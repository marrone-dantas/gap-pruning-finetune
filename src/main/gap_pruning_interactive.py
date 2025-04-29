import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_pruning as tp
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import uuid # Used for unique filenames in visualization buffer
import copy

# Configure module-level logger
global_logger = logging.getLogger(__name__)
log = logging.getLogger(__name__)
# Set logging level and format
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class GapPruningInteractive:
    """
    Performs channel pruning on Conv2d layers based on the spatial statistics
    (mean, max, std deviation) of activations across different classes.

    It calculates the standard deviation of these per-filter spatial statistics
    across classes to determine filter importance. Less important filters
    (those with low std deviation across class means) are considered for pruning.

    An adaptive layer-wise pruning rate is determined based on layer sensitivity
    (sum of std deviations) and parameter count, aiming for a global pruning target.

    Additionally, it applies a correlation-based weighting, giving higher
    importance to filters that are less correlated with others within a layer,
    potentially preserving more diverse features.
    """

    class _Hook:
        """
        Internal forward hook to accumulate multiple spatial statistics (mean, max, std)
        for each filter, separated by class label. Attached to Conv2d layers during
        the statistics computation phase.
        """
        def __init__(self, parent: 'GapPruningInteractive', layer_name: str) -> None:
            """
            Initializes the hook.

            Args:
                parent: The parent `GapPruningInteractive` instance.
                layer_name: The name of the layer this hook is attached to.
            """
            self.parent = parent
            self.layer_name = layer_name

        def __call__(self, module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            """
            The hook function called during the forward pass. Accumulates spatial stats.

            Args:
                module: The layer module.
                _inp: The input tensor to the module (unused).
                output: The output tensor from the module (activations).
            """
            # Get the current batch's labels from the parent instance
            labels = self.parent.current_labels
            # Check for label/activation size mismatch or if labels are not set
            if labels is None or output.size(0) != len(labels):
                global_logger.debug(
                    f"Skipping layer {self.layer_name} hook: label/activation size mismatch or labels not set."
                )
                return

            # Compute per-filter spatial statistics over spatial dimensions (H, W)
            # mean_pool: shape (batch_size, num_filters)
            # max_pool: shape (batch_size, num_filters)
            # std_pool: shape (batch_size, num_filters)
            mean_pool = output.mean(dim=[2, 3])
            max_pool = output.amax(dim=[2, 3])
            # std_pool might return NaN if activation is constant across spatial dims; handle it
            std_pool = torch.std(output, dim=[2, 3])
            std_pool = torch.nan_to_num(std_pool, nan=0.0)

            # Initialize storage per layer and per class if not already present
            layer_stats = self.parent.accumulated_stats.setdefault(self.layer_name, {})

            # Accumulate stats for each sample in the batch, grouped by class
            for i, cls in enumerate(labels):
                cls = int(cls) # Ensure class label is integer
                cls_dict = layer_stats.setdefault(cls, {
                    'mean_sum': torch.zeros_like(mean_pool[0]),
                    'max_sum' : torch.zeros_like(max_pool[0]),
                    'std_sum' : torch.zeros_like(std_pool[0]),
                })
                # Sum up the stats for each filter within the same class
                cls_dict['mean_sum'] += mean_pool[i]
                cls_dict['max_sum']  += max_pool[i]
                cls_dict['std_sum']  += std_pool[i]

    def __init__(
        self,
        model: nn.Module,
        dataset: Any, # Can be a Dataset or DataLoader
        batch_size: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Initializes the pruning utility.

        Args:
            model: Trained nn.Module to prune. Must be on the specified device.
            dataset: Dataset or DataLoader yielding (input, label) pairs.
                     Used for collecting activation statistics.
            batch_size: Effective batch size for stats collection if `dataset` is not a DataLoader.
                        If `dataset` is a DataLoader, its batch size is used.
            device: Computation device ('cuda' or 'cpu'). Model and tensors will be moved here.
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        # Dictionary to store the pruning proposal {layer_name: [list_of_pruned_indices]}
        self.pruning_proposal: Optional[Dict[str, List[int]]] = None

        # State variables for on-the-fly statistics accumulation
        # accumulated_stats: {layer_name: {class_id: {stat_name: sum_of_stats_tensor}}}
        self.accumulated_stats: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        # Stores the final calculated standard deviations of class means for each filter
        # std_devs: {layer_name: tensor_of_std_devs_per_filter}
        self.std_devs: Dict[str, torch.Tensor] = {}
        # Holds the labels for the current batch being processed by hooks
        self.current_labels: Optional[List[int]] = None
        # List of hook handles to manage their removal
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        global_logger.info(f"Initialized GapPruningInteractive on device: {self.device}")

    def _register_hooks(self) -> None:
        """
        Attach forward hooks (`_Hook` instances) to all Conv2d layers in the model
        to collect activation statistics. Removes any existing hooks first.
        """
        self._remove_hooks() # Ensure no hooks are already attached
        self.accumulated_stats.clear() # Clear previous stats
        # self.class_sample_counts is not used directly in this refined version
        # as balanced sampling is handled in compute_stats_on_the_fly

        # Iterate through all modules and register hooks on Conv2d layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Register the hook and store the handle for later removal
                handle = module.register_forward_hook(
                    self._Hook(self, name)
                )
                self._hook_handles.append(handle)
                # global_logger.debug(f"Registered hook on layer: {name}") # Too verbose
        global_logger.info(f"Registered hooks on {len(self._hook_handles)} Conv2d layers.")

    def _remove_hooks(self) -> None:
        """
        Remove any existing forward hooks managed by this instance.
        """
        if self._hook_handles:
            for h in self._hook_handles:
                h.remove() # Remove the hook
            self._hook_handles.clear() # Clear the list of handles
            global_logger.info("Removed all registered hooks.")

    def compute_stats_on_the_fly(
        self,
        sample_limit: Optional[int] = None,
        file_path: str = 'std_devs_of_class_means.pth',
        load: bool = False
    ) -> None:
        """
        Computes per-filter standard deviation of spatial statistics (mean, max, std)
        aggregated across classes. Only uses samples the model predicts correctly.
        Balancing is applied to process a similar number of samples per class.
        Results are saved to or loaded from a file.

        The process involves two phases:
        1. Count samples per class in the entire dataset to determine the minimum class size.
        2. Iterate through the dataset again, collecting activations for correctly predicted
           samples, limited by the minimum class size or `sample_limit`, per class.

        Args:
            sample_limit: Optional cap on the *total* number of correct samples processed
                          across all classes. The per-class target will be
                          min(min_class_size, sample_limit // num_classes).
            file_path: Path to load/save the computed `self.std_devs`.
            load: If True, attempt to load stats from `file_path`. If loading fails
                  or load=False, computation proceeds.
        """
        # Attempt to load existing stats if requested and file exists
        if load and os.path.isfile(file_path):
            try:
                self.std_devs = torch.load(file_path, map_location='cpu') # Load to CPU
                # Ensure tensors are on the correct device if needed later
                self.std_devs = {k: v.to(self.device) for k, v in self.std_devs.items()}
                global_logger.info(f"Loaded std_devs from {file_path}")
                # We also need accumulated_stats for correlation weighting if loading
                # A more robust implementation might save/load accumulated_stats too.
                # For now, if loading std_devs, we assume correlation weighting was
                # already applied during the saving process or skip it.
                # Let's recompute accumulated_stats lightly to enable weighting logic,
                # but this is a potential area for improvement (saving accumulated_stats)
                global_logger.warning("Loaded stats, but recomputing accumulated_stats for correlation weighting. Consider saving accumulated_stats for faster loading.")
                # Quick pass to populate accumulated_stats minimally
                # This is a workaround; better to save/load accumulated_stats
                self._register_hooks()
                # Use a very small number of samples or a dummy pass just to populate
                # the keys in accumulated_stats if possible, or restructure
                # _apply_correlation_weights to work directly with std_devs or load more data.
                # A simpler approach: if loaded, skip correlation weighting here unless
                # accumulated_stats are also loaded. Given the current structure,
                # we need accumulated_stats for _apply_correlation_weights.
                # Let's proceed with recomputing if loading doesn't include accumulated_stats.
                # The current structure implies accumulated_stats is needed for weighting,
                # which happens *after* initial std_dev computation from accumulated_stats.
                # So, if we load std_devs, we probably shouldn't re-apply weighting based
                # on potentially outdated or missing accumulated_stats.
                # Let's refine: If loading, skip recomputing and weighting.
                return # Exit after successful loading

            except Exception as e:
                global_logger.warning(f"Failed to load stats from {file_path}: {e}. Recomputing.")
                # Clear std_devs if loading failed
                self.std_devs.clear()

        # Prepare DataLoader if dataset is not already one
        if isinstance(self.dataset, DataLoader):
            loader = self.dataset
        else:
            loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False, # No need to shuffle for stats collection
                num_workers=max(1, os.cpu_count() // 2) # Use multiple workers
            )

        # PHASE 1: Count all samples per class in the dataset
        global_logger.info("Starting phase 1: Counting class samples...")
        class_counts: Dict[int, int] = {}
        # Iterate through the loader to count class occurrences
        for _, labels in tqdm(loader, desc='Counting class samples'):
            unique, counts = torch.unique(labels, return_counts=True)
            for cls, cnt in zip(unique.tolist(), counts.tolist()):
                class_counts[cls] = class_counts.get(cls, 0) + cnt

        if not class_counts:
            raise ValueError("No class samples found in the dataset.")

        min_count = min(class_counts.values())
        num_classes = len(class_counts)
        # Determine the target number of *correctly predicted* samples per class
        target_per_class = min_count
        if sample_limit is not None:
            # Ensure the total target across classes doesn't exceed sample_limit
            target_per_class = min(min_count, sample_limit // num_classes)

        global_logger.info(
            f"Phase 1 complete. Found {num_classes} classes. "
            f"Targeting ~{target_per_class} correctly predicted samples per class."
        )

        # PHASE 2: Collect balanced correct samples and accumulate activations
        self._register_hooks() # Attach hooks before the forward pass
        per_class_seen_correct: Dict[int, int] = {cls: 0 for cls in class_counts}
        total_seen_correct = 0
        self.model.eval() # Set model to evaluation mode
        global_logger.info("Starting phase 2: Collecting balanced correct activations...")

        with torch.no_grad(): # Disable gradient calculation
            # Iterate through the loader to process samples
            for inputs, labels in tqdm(loader, desc='Collecting balanced correct activations'):
                # Move inputs to the specified device
                inputs_device = inputs.to(self.device)
                # Convert labels to CPU list for easy indexing
                labels_cpu = labels.cpu().tolist()

                # --- First pass: get predictions to identify correct samples ---
                # Temporarily disable label setting to prevent hook accumulation
                # during this prediction-only pass.
                self.current_labels = None
                outputs = self.model(inputs_device)
                # Get predicted class indices
                preds = outputs.argmax(dim=1).cpu().tolist()

                # Identify indices of correctly predicted samples
                correct_idx_in_batch = [
                    i for i, (pred, label) in enumerate(zip(preds, labels_cpu)) if pred == label
                ]

                if not correct_idx_in_batch:
                    # No correct predictions in this batch, skip to the next
                    continue

                # --- Second pass: select correct samples within per-class quota and accumulate ---
                keep_idx_for_accumulation = []
                # Filter correct samples that are still needed to reach the target per class
                for i in correct_idx_in_batch:
                    current_class = labels_cpu[i]
                    if per_class_seen_correct[current_class] < target_per_class:
                        keep_idx_for_accumulation.append(i)
                        # Increment the count for this class
                        per_class_seen_correct[current_class] += 1

                if not keep_idx_for_accumulation:
                    # No samples from this batch are needed for accumulation, skip
                    continue

                # Set labels for the samples that will be processed by the hooks
                self.current_labels = [labels_cpu[i] for i in keep_idx_for_accumulation]

                # Perform forward pass ONLY for the selected correct samples
                # The hooks will be triggered and accumulate stats for these samples
                _ = self.model(inputs_device[keep_idx_for_accumulation])

                # Update the total count of samples processed for accumulation
                total_seen_correct += len(keep_idx_for_accumulation)

                # Check stopping criteria
                # Stop if we have reached the sample_limit (if specified)
                if sample_limit is not None and total_seen_correct >= sample_limit:
                    global_logger.info(f"Reached total sample limit of {sample_limit}. Stopping data collection.")
                    break
                # Stop if we have collected the target number of samples for all classes
                if all(count >= target_per_class for count in per_class_seen_correct.values()):
                     global_logger.info(f"Collected target samples ({target_per_class} per class) for all classes. Stopping data collection.")
                     break

        self._remove_hooks() # Remove hooks after data collection
        self.current_labels = None # Clear current labels

        global_logger.info("Finished collecting balanced correct activations. Computing std deviations...")

        # Compute the average stats per class and then the standard deviation across classes
        self.std_devs.clear()
        for layer, stats_by_class in tqdm(self.accumulated_stats.items(), desc='Computing stats per layer'):
            stat_stds = [] # List to hold std dev for each stat type (mean, max, std)
            # Check if we collected enough data for this layer/classes
            # A layer might not have received 'target_per_class' samples if the loop broke early
            # Find the actual number of samples processed for each class that reached this layer
            # This assumes that all samples passed through all hooked layers
            # A more robust check would be to store per-layer per-class counts.
            # Using target_per_class as a proxy here, but be aware of potential inaccuracies
            # if early stopping occurred and some classes/layers didn't get full data.
            effective_samples_per_class = target_per_class # Simplification

            # Compute average stats per class and collect them for std deviation calculation
            per_class_mean_sum = []
            per_class_max_sum = []
            per_class_std_sum = []
            for cls in sorted(stats_by_class.keys()): # Sort to ensure consistent order
                 cls_stats = stats_by_class[cls]
                 # Only include classes for which we processed samples
                 # This relies on the fact that accumulated_stats only has entries for classes processed
                 # and assumes each class processed contributed `effective_samples_per_class` samples.
                 # If not all classes were processed up to `target_per_class` due to early stopping,
                 # this calculation will be based on fewer samples for those classes.
                 per_class_mean_sum.append(cls_stats['mean_sum'] / effective_samples_per_class)
                 per_class_max_sum.append(cls_stats['max_sum'] / effective_samples_per_class)
                 per_class_std_sum.append(cls_stats['std_sum'] / effective_samples_per_class)


            # Calculate the standard deviation across classes for each stat type
            if per_class_mean_sum: # Only compute if there's data
                stacked_means = torch.stack(per_class_mean_sum, dim=0).cpu()
                stat_stds.append(torch.std(stacked_means, dim=0))

            if per_class_max_sum:
                stacked_maxes = torch.stack(per_class_max_sum, dim=0).cpu()
                stat_stds.append(torch.std(stacked_maxes, dim=0))

            if per_class_std_sum:
                stacked_stds_of_stds = torch.stack(per_class_std_sum, dim=0).cpu()
                stat_stds.append(torch.std(stacked_stds_of_stds, dim=0))

            # Combine the standard deviations (e.g., by averaging) and move to device
            if stat_stds:
                # Calculate the average standard deviation across the different stats
                combined_std = sum(stat_stds) / len(stat_stds)
                self.std_devs[layer] = combined_std.to(self.device)
            else:
                 global_logger.warning(f"No stats computed for layer {layer}. Skipping.")
                 self.std_devs[layer] = torch.zeros(self.get_layer_by_name(layer).out_channels).to(self.device) # Add a zero tensor as placeholder

        # Apply clustering weights to adjust std_devs based on filter correlation
        if self.accumulated_stats and target_per_class > 0: # Ensure we have data and samples were processed
            self._apply_correlation_weights(target_per_class)
        else:
            global_logger.warning("Skipping correlation weighting: no accumulated stats or target_per_class is zero.")


        # Save computed stats to file
        try:
            # Save to CPU to avoid device mismatch when loading
            std_devs_cpu = {k: v.cpu() for k, v in self.std_devs.items()}
            torch.save(std_devs_cpu, file_path)
            global_logger.info(f"Saved std_devs to {file_path}")
        except Exception as e:
            global_logger.warning(f"Failed to save stats to {file_path}: {e}")

    def _apply_correlation_weights(self, samples_per_class: int) -> None:
        """
        Weights each filter's standard deviation inversely to the size of its
        cluster. Filters that are highly correlated (and thus in larger clusters)
        get a smaller weight, making them relatively less important for pruning
        compared to filters in smaller clusters. This aims to preserve unique features.

        Clustering is based on the per-class mean activations for each filter.

        Args:
            samples_per_class: The number of samples used to compute the class means.
                               Used for normalizing the accumulated sums.
        """
        global_logger.info("Applying correlation weights based on clustering...")
        for layer, stats_by_class in tqdm(self.accumulated_stats.items(), desc='Clustering and weighting filters'):
            # Get sorted class keys to ensure consistent ordering
            cls_keys = sorted(stats_by_class.keys())

            if not cls_keys:
                 global_logger.warning(f"No class stats for layer {layer}. Skipping clustering.")
                 continue

            # Prepare features for clustering: per-filter mean activation across classes
            # Shape will be (num_classes, num_filters)
            per_class_means = [
                stats_by_class[cls]['mean_sum'] / samples_per_class
                for cls in cls_keys
            ]

            if not per_class_means:
                 global_logger.warning(f"Per-class means empty for layer {layer}. Skipping clustering.")
                 continue

            # Stack and transpose to get shape (num_filters, num_classes) for clustering
            # Move to CPU and convert to NumPy for scikit-learn
            features = torch.stack(per_class_means, dim=0).cpu().numpy().T

            n_filters = features.shape[0]
            n_classes = features.shape[1]

            if n_filters < 2 or n_classes < 2:
                 global_logger.warning(
                     f"Insufficient filters ({n_filters}) or classes ({n_classes}) "
                     f"for clustering in layer {layer}. Skipping."
                 )
                 continue

            # Determine the number of clusters. Use a heuristic: at least 2,
            # up to a tenth of filters or number of classes, whichever is smaller.
            n_clusters = max(2, min(n_filters // 10, n_classes))

            try:
                # Perform Agglomerative Clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(features)

                # Calculate the size of each cluster
                sizes = Counter(labels)

                # Compute weights inversely proportional to cluster size
                # Filters in larger clusters get smaller weights
                weights = np.array([1.0 / sizes[label] for label in labels], dtype=np.float32)

                # Convert weights to a tensor and move to the correct device
                weight_tensor = torch.from_numpy(weights).to(self.device)

                # Apply the weights to the standard deviations
                # Ensure the weight tensor size matches the std_devs tensor size for the layer
                if layer in self.std_devs and self.std_devs[layer].size(0) == weight_tensor.size(0):
                    self.std_devs[layer] = self.std_devs[layer] * weight_tensor
                else:
                    global_logger.warning(
                        f"Mismatch between std_devs size ({self.std_devs.get(layer, torch.empty(0)).size(0)}) "
                        f"and weight_tensor size ({weight_tensor.size(0)}) for layer {layer}. "
                        "Could not apply correlation weights."
                    )
            except Exception as e:
                 global_logger.warning(f"Clustering failed for layer {layer}: {e}. Skipping correlation weighting for this layer.")


    def _allocate_layer_prune_rates(self, global_prune_rate: float) -> Dict[str, float]:
        """
        Allocates a specific pruning rate to each prunable layer.
        The rate for each layer is determined based on its 'sensitivity' (sum of
        per-filter std deviations) and parameter count, aiming to achieve a
        target global pruning percentage across all prunable layers.

        The logic is to prioritize pruning filters in layers that are less sensitive
        (lower std deviations), while also considering the size of the layer.

        Args:
            global_prune_rate: The desired overall percentage of parameters to prune
                               across the entire model (as a float between 0.0 and 1.0).

        Returns:
            A dictionary mapping layer names to their calculated pruning rates (float).
        """
        global_logger.info(f"Allocating layer-wise prune rates for global target {global_prune_rate:.3f}...")

        # Compute sensitivity-based importance for each layer
        # Sensitivity is the sum of the weighted std deviations for all filters in the layer
        sensitivities = {
            l: v.sum().item() for l, v in self.std_devs.items() if v.numel() > 0
        }
        # Handle case where all std_devs are zero or layers have no filters
        total_sens = sum(sensitivities.values())
        if total_sens == 0:
            global_logger.warning("Total sensitivity is zero. Cannot allocate rates based on sensitivity.")
            # Fallback: assign uniform rate or 0 if global_prune_rate is 0
            if global_prune_rate > 0:
                 # Fallback to uniform distribution if no sensitivity info
                 num_prunable_layers = len(self.std_devs)
                 if num_prunable_layers > 0:
                     uniform_rate = global_prune_rate / num_prunable_layers # This is not parameter weighted
                     global_logger.warning(f"Using uniform rate fallback: {uniform_rate:.3f} per layer.")
                     return {l: uniform_rate for l in self.std_devs.keys()}
                 else:
                      global_logger.warning("No prunable layers found.")
                      return {}
            else:
                 return {l: 0.0 for l in self.std_devs.keys()} # If target is 0, all rates are 0

        # Normalize sensitivities to get importance scores (proportion of total sensitivity)
        importance = {l: sens / total_sens for l, sens in sensitivities.items()}

        # Compute parameter count for each layer that we have std_devs for
        param_counts: Dict[str, int] = {}
        for layer_name in self.std_devs.keys():
            layer = self.get_layer_by_name(layer_name)
            # Check if the layer exists and has a weight parameter (typical for Conv2d)
            if layer is not None and hasattr(layer, 'weight') and layer.weight is not None:
                param_counts[layer_name] = layer.weight.numel()
            else:
                 global_logger.warning(f"Could not get parameter count for layer {layer_name}. Skipping.")
                 param_counts[layer_name] = 0 # Assign 0 if parameter count cannot be determined

        total_params = sum(param_counts.values()) or 1 # Prevent division by zero

        # Calculate the weighted sum in the denominator: sum_i [ (1 - importance_i) * param_proportion_i ]
        weighted_sum_denominator = sum(
            (1.0 - importance.get(l, 0.0)) * (param_counts.get(l, 0) / total_params)
            for l in self.std_devs.keys() # Iterate over all layers for which we have std_devs
        )

        # Calculate beta. Handle division by zero if the weighted sum is zero.
        beta = (global_prune_rate / weighted_sum_denominator) if weighted_sum_denominator > 1e-9 else 0.0
        # A small epsilon is added to the denominator to handle near-zero values gracefully.

        # Calculate the final layer-wise rates
        rates: Dict[str, float] = {}
        for layer_name in self.std_devs.keys():
            # Get the importance for this layer, default to 0 if not found (though it should be there)
            imp = importance.get(layer_name, 0.0)
            # Calculate the raw rate using beta and (1 - importance)
            raw_rate = (1.0 - imp) * beta
            # Clamp the rate to be between 0.0 and 1.0
            rates[layer_name] = float(min(max(raw_rate, 0.0), 1.0))

        # Log the calculated rates
        global_logger.info("Calculated layer-wise prune rates:")
        for layer_name, rate in rates.items():
            global_logger.info(f"  {layer_name}: {rate:.4f}")

        return rates

    def find_filter_similarity_mapping(
        self,
        orig_weight: torch.Tensor,
        pruned_weight: torch.Tensor
    ) -> Dict[int, int]:
        """
        Given the original layer weight tensor and the weight tensor after pruning
        (where filters have been removed), this function finds the most similar
        remaining filter in the pruned tensor for each filter that existed in the
        original tensor. This mapping is useful for transferring information or
        understanding the relationship between original and pruned filters.

        Similarity is measured by L2 distance between flattened filter tensors.

        Args:
            orig_weight: The weight tensor of the layer before pruning.
                         Expected shape: (out_channels, in_channels, kH, kW)
            pruned_weight: The weight tensor of the layer after pruning.
                           Expected shape: (new_out_channels, in_channels, kH, kW)

        Returns:
            A dictionary where keys are the indices of filters in the original
            `orig_weight` tensor and values are the indices of the most similar
            filters in the `pruned_weight` tensor. Returns an empty dictionary
            if shapes are incompatible or input tensors are empty.
        """
        # Validate input tensor shapes
        if (orig_weight.ndim != 4 or pruned_weight.ndim != 4 or
            orig_weight.shape[1:] != pruned_weight.shape[1:]):
            global_logger.warning(
                f"Mismatch shapes for similarity mapping: {orig_weight.shape} vs {pruned_weight.shape}. "
                "Expected (out_channels, in_channels, kH, kW)."
            )
            return {}

        # Handle empty tensors
        if orig_weight.size(0) == 0 or pruned_weight.size(0) == 0:
             global_logger.warning("Original or pruned weight tensor is empty. Cannot perform similarity mapping.")
             return {}

        # Flatten the filters into vectors: shape (num_filters, filter_size)
        o_flat = orig_weight.view(orig_weight.size(0), -1).cpu()
        p_flat = pruned_weight.view(pruned_weight.size(0), -1).cpu()

        # Compute pairwise L2 distances between all original and pruned filters
        # Using torch.cdist for efficient batch computation
        # dists shape: (num_original_filters, num_pruned_filters)
        dists = torch.cdist(o_flat, p_flat)

        # Find the index of the minimum distance for each original filter
        # matches shape: (num_original_filters)
        matches = torch.argmin(dists, dim=1)

        # Create the mapping dictionary
        # Map original filter index to the index of the most similar pruned filter
        similarity_map = {i: int(matches[i]) for i in range(matches.size(0))}

        return similarity_map

    def prune_online(
        self,
        global_prune_rate: float = 0.5,
        example_input_size: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        """
        Applies channel pruning to the model's Conv2d layers based on the
        computed `self.std_devs`. Pruning is performed layer by layer with
        adaptive rates calculated by `_allocate_layer_prune_rates`.

        The pruning process uses `torch_pruning` to handle dependencies.

        Args:
            global_prune_rate: The soft target proportion of total model parameters
                               to prune (0.0 to 1.0).
            example_input_size: A tuple representing the shape of a dummy input
                                tensor required by `torch_pruning` to build the
                                dependency graph.

        Returns:
            A tuple containing:
            - A pandas DataFrame summarizing the pruning results for each layer,
              with columns: ['layer', 'original', 'pruned', 'remaining'].
            - A dictionary `pruned_data` mapping layer names to a list of indices
              of the filters that were pruned from that layer. This is stored
              internally in `self.pruning_proposal`.

        Raises:
            RuntimeError: If `compute_stats_on_the_fly()` has not been called
                          and `self.std_devs` is empty.
        """
        
        
        
        # Ensure statistics have been computed
        if not self.std_devs:
            raise RuntimeError(
                "Statistics not computed. Run compute_stats_on_the_fly() first "
                "to populate self.std_devs."
            )

        global_logger.info(f"Starting online pruning with global target rate: {global_prune_rate:.3f}")

        # Dictionary to store the indices of pruned filters for each layer
        self.pruning_proposal = {} # Reset the proposal
        pruned_data = {} # Alias for clarity
        full_pruned_data = {} # Alias for clarity

        # Log total parameters before pruning
        total_params_before = sum(p.numel() for p in self.model.parameters())
        global_logger.info(f"Total parameters before pruning: {total_params_before}")

        # Build the dependency graph using torch_pruning.
        # This typically needs the model on CPU to avoid memory issues,
        # especially for large models or limited GPU memory.
        original_device = next(self.model.parameters()).device # Get current device
        self.model.to('cpu')
        global_logger.info("Building dependency graph on CPU...")
        try:
            # Create a dummy input tensor on CPU with the specified size
            dummy_input = torch.randn(*example_input_size).to('cpu')
            # Build the dependency graph
            DG = tp.DependencyGraph().build_dependency(
                self.model, example_inputs=dummy_input
            )
            global_logger.info("Dependency graph built successfully.")
        except Exception as e:
             global_logger.error(f"Failed to build dependency graph: {e}")
             # Move model back to original device before raising
             self.model.to(original_device)
             raise

        # Move the model back to its original device
        self.model.to(original_device)
        global_logger.info(f"Model moved back to {original_device}.")


        # Calculate layer-wise pruning rates based on sensitivity and parameter count
        rates = self._allocate_layer_prune_rates(global_prune_rate)

        # List to store report data for the DataFrame
        report: List[Dict[str, Any]] = []

        self.model.eval() # Set model to evaluation mode

        # Iterate through layers that have computed standard deviations
        for layer_name, std in tqdm(self.std_devs.items(), desc='Pruning layers'):
            layer = self.get_layer_by_name(layer_name)
            # Check if the layer exists and is a Conv2d layer (prunable type)
            if layer is None or not isinstance(layer, nn.Conv2d):
                global_logger.warning(f"Layer '{layer_name}' not found or not a Conv2d layer. Skipping pruning.")
                continue

            # Get the original number of output channels (filters)
            original_out_channels = layer.weight.size(0)

            # If there are no std_devs calculated for this layer's filters, skip pruning
            if std.numel() == 0:
                 global_logger.warning(f"No valid std_devs for layer {layer_name}. Skipping pruning.")
                 report.append({
                    'layer': layer_name,
                    'original': original_out_channels,
                    'pruned': 0,
                    'remaining': original_out_channels
                 })
                 continue


            # Before pruning, the current layer.weight corresponds to the original filters.
            # We need to map the std_devs (which were potentially weighted based on original filters)
            # to the *current* state of the layer if it has already been pruned by other means
            # (e.g., previous pruning steps if this were part of an iterative process).
            # The current code assumes a single pruning pass on an initial model.
            # In a multi-pass scenario, `find_filter_similarity_mapping` would be used
            # to map std_devs from the original filter space to the current filter space.
            # For this single pass, we can directly use the std_devs assuming they
            # correspond to the current filters (which are the original filters initially).

            # Get the pruning rate for this layer
            rate = rates.get(layer_name, 0.0) # Default to 0.0 if rate is not allocated
            if rate == 0.0:
                 global_logger.info(f"Pruning rate for {layer_name} is 0. Skipping pruning.")
                 report.append({
                    'layer': layer_name,
                    'original': original_out_channels,
                    'pruned': 0,
                    'remaining': original_out_channels
                 })
                 continue

            # Determine the number of filters to prune for this layer
            total_filters = original_out_channels
            to_prune_count = int(total_filters * rate)
            # Ensure we don't prune more filters than available or leave zero filters
            to_prune_count = max(0, min(to_prune_count, total_filters - 1)) # Always keep at least one filter
            if to_prune_count == 0:
                global_logger.info(f"Calculated prune count for {layer_name} is 0. Skipping pruning.")
                report.append({
                    'layer': layer_name,
                    'original': original_out_channels,
                    'pruned': 0,
                    'remaining': original_out_channels
                })
                continue

            # Identify the indices of the filters to prune.
            # Filters with the lowest standard deviations (least variation across classes)
            # are considered less important and are selected for pruning.
            # torch.argsort returns indices that would sort the tensor.
            # We take the first `to_prune_count` indices which correspond to the smallest std_devs.
            # Move std to cpu for argsort if needed, but should be on device. Keep it on device.
            prune_indices = torch.argsort(std).tolist()
            
            # Store the indices of pruned filters
            pruned_data[layer_name]      = copy.deepcopy(prune_indices[:to_prune_count])
            full_pruned_data[layer_name] = copy.deepcopy(prune_indices)

            # Get the pruning plan from the dependency graph
            # This plan includes removing the specified filters and updating connected layers
            try:
                 plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=pruned_data[layer_name])
                 # Execute the pruning plan
                 plan.exec()
                 global_logger.info(
                     f"Pruned {to_prune_count}/{original_out_channels} filters "
                     f"from layer {layer_name} (rate: {rate:.4f})."
                 )
            except Exception as e:
                 global_logger.error(f"Error executing pruning plan for layer {layer_name}: {e}. Skipping this layer.")
                 # Revert the entry in pruned_data if pruning failed
                 if layer_name in pruned_data:
                     del pruned_data[layer_name]
                     del full_pruned_data[layer_name]
                 # Add a report entry indicating failure or no pruning
                 report.append({
                    'layer': layer_name,
                    'original': original_out_channels,
                    'pruned': 0, # Report 0 pruned if execution failed
                    'remaining': original_out_channels
                 })
                 continue


            # Get the number of remaining filters after pruning
            remaining_out_channels = layer.weight.size(0)

            # Add pruning results to the report
            report.append({
                'layer': layer_name,
                'original': original_out_channels,
                'pruned': to_prune_count,
                'remaining': remaining_out_channels
            })


        # Log total parameters after pruning
        total_params_after = sum(p.numel() for p in self.model.parameters())
        global_logger.info(f"Total parameters after pruning: {total_params_after}")
        pruning_percentage = (total_params_before - total_params_after) / total_params_before * 100 if total_params_before > 0 else 0
        global_logger.info(f"Total parameters pruned: {total_params_before - total_params_after} ({pruning_percentage:.2f}%)")

        # Create a pandas DataFrame from the report
        pruning_report_df = pd.DataFrame(report)

        # Lets check if we create a valid model
        

        return pruning_report_df, pruned_data, full_pruned_data


    def global_pruning(
        self,
        global_prune_rate: float = 0.5,
        example_input_size: Tuple[int, ...] = (1, 3, 224, 224)
    ) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        """
        Performs global channel pruning by ranking **all** filters in the network
        by their std-dev and removing the lowest‐variance ones until
        global_prune_rate * total_filters have been pruned.

        Returns:
            - pruning_report_df: DataFrame with columns ['layer','original','pruned','remaining']
            - pruned_data: dict mapping layer_name -> list of pruned filter indices
        """
        def normalize_to_unit(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            """
            Normaliza os valores de `tensor` para o intervalo [0, 1].

            Args:
                tensor: Tensor qualquer.
                eps: Pequeno valor para evitar divisão por zero se max == min.

            Returns:
                Tensor normalizado em [0,1].
            """
            min_val = tensor.min()
            max_val = tensor.max()
            return (tensor - min_val) / (max_val - min_val + eps)
        
        if not self.std_devs:
            raise RuntimeError(
                "No statistics found. Run compute_stats_on_the_fly() first."
            )

        global_logger.info(f"Starting GLOBAL pruning at {global_prune_rate:.2%} target...")
        total_before = sum(p.numel() for p in self.model.parameters())

        # 1) Build dependency graph on CPU
        original_device = next(self.model.parameters()).device
        self.model.to('cpu')
        dummy = torch.randn(*example_input_size)
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=dummy)
        self.model.to(original_device)

        # 2) Flatten all filters into one list: (layer_name, filter_idx, std_val)
        all_filters: List[Tuple[str,int,float]] = []
        for layer_name, std in self.std_devs.items():
            std_norm = normalize_to_unit(std)
            # skip empty or non-Conv layers
            if std.numel() == 0:
                continue
            for idx, val in enumerate(std_norm.tolist()):
                all_filters.append((layer_name, idx, val))

        total_filters = len(all_filters)
        num_to_prune   = int(global_prune_rate * total_filters)
        global_logger.info(f"Total filters = {total_filters}, pruning {num_to_prune} filters.")

        if num_to_prune <= 0:
            global_logger.info("Nothing to prune at this rate.")
            return pd.DataFrame(columns=['layer','original','pruned','remaining']), {}

        # 3) Sort globally and pick the lowest‐std filters
        all_filters.sort(key=lambda x: x[2])  # ascending std
        to_prune = all_filters[:num_to_prune]

        # 4) Group by layer
        pruned_data: Dict[str,List[int]] = {}
        for layer_name, idx, _ in to_prune:
            pruned_data.setdefault(layer_name, []).append(idx)

        # 5) Apply pruning per layer
        report = []
        self.pruning_proposal = {}
        for layer_name, idxs in tqdm(pruned_data.items(), desc="Global pruning"):
            layer = self.get_layer_by_name(layer_name)
            if layer is None or not isinstance(layer, torch.nn.Conv2d):
                global_logger.warning(f"Skipping {layer_name}: not a Conv2d.")
                continue

            original_out = layer.weight.size(0)
            # make sure we never try to prune all filters
            idxs = [i for i in sorted(set(idxs)) if i < original_out]
            idxs = idxs[: max(0, original_out - 1)]  

            if not idxs:
                report.append({
                    'layer': layer_name,
                    'original': original_out,
                    'pruned': 0,
                    'remaining': original_out
                })
                continue

            # execute prune
            try:
                plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=idxs)
                plan.exec()
                self.pruning_proposal[layer_name] = copy.deepcopy(idxs)
                pruned_count = len(idxs)
                remaining = layer.weight.size(0)
                report.append({
                    'layer': layer_name,
                    'original': original_out,
                    'pruned': pruned_count,
                    'remaining': remaining
                })
                global_logger.info(
                    f"Pruned {pruned_count}/{original_out} from {layer_name}"
                )
            except Exception as e:
                global_logger.error(f"Failed to prune {layer_name}: {e}")
                report.append({
                    'layer': layer_name,
                    'original': original_out,
                    'pruned': 0,
                    'remaining': original_out
                })

        # 6) Summary DataFrame
        pruning_report_df = pd.DataFrame(report)

        total_after  = sum(p.numel() for p in self.model.parameters())
        global_logger.info(
            f"Params before: {total_before}, after: {total_after}, "
            f"pruned {(total_before-total_after)} ({(total_before-total_after)/total_before:.2%})"
        )

        return pruning_report_df, pruned_data

    def visualize_gradients_pruned_vs_kept(
        self,
        layers: Optional[List[str]] = None,
        use_loader: bool = True,
        num_samples: int = 3,
        random_sampling: bool = True,
        sample_inputs: Optional[List[torch.Tensor]] = None,
        denormalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        buffer: bool = False,
        buffer_path: str = './gradient_visualization_buffers', # Different default path
        buffer_prefix: str = 'grad_vis',
        just_save_data: bool = False,
        target_class_for_grad: Optional[int] = None # Added option to target a specific class
    ) -> None:
        """
        Visualizes the gradients of pruned vs. kept filters with respect to the
        predicted class score (or a specified target class score) for given
        input samples. This shows which spatial regions within the filter's
        feature map influenced the final prediction score.

        Args:
            layers: A list of specific layer names (strings) to visualize gradients for.
                    Defaults to all layers that were pruned according to `self.pruning_proposal`
                    and are Conv2d layers.
            use_loader: If True, draw input samples from `self.dataset` DataLoader.
                        If False, `sample_inputs` must be provided.
            num_samples: The number of input samples to visualize.
            random_sampling: If `use_loader` is True, randomly sample `num_samples`
                             from the dataset. If False, take the first `num_samples`.
            sample_inputs: A list of input tensors to use for visualization when
                           `use_loader` is False.
            denormalize: If True, attempt to reverse the normalization for displaying the input image.
            mean: List of mean values used for normalization (for denormalization).
            std: List of standard deviation values used for normalization (for denormalization).
            buffer: If True, save the generated visualization figures to disk.
            buffer_path: Directory to save the gradient visualization images if `buffer` is True.
                         Created if it does not exist.
            buffer_prefix: A string prefix for the filenames of saved images.
            just_save_data: If True, save the figures to `buffer_path` but do not display them.
            target_class_for_grad: If specified (an integer class index), compute gradients
                                   with respect to the score of this specific class.
                                   Otherwise, compute gradients with respect to the score
                                   of the model's predicted class for the input.
        """
        # Ensure pruning has been performed and a proposal exists
        if self.pruning_proposal is None or not self.pruning_proposal:
            global_logger.warning("No pruning proposal found. Run prune_online() first.")
            return

        # Select and validate layers for gradient visualization
        if layers is None:
            # Default to all layers that were pruned and are Conv2d
            layers_to_visualize = [
                l for l in self.pruning_proposal.keys()
                if self.pruning_proposal[l] and isinstance(self.get_layer_by_name(l), nn.Conv2d)
            ]
            if not layers_to_visualize:
                global_logger.warning("No pruned Conv2d layers found in the pruning proposal. Cannot visualize gradients.")
                return
            global_logger.info(f"Visualizing gradients for pruned Conv2d layers: {layers_to_visualize}")

        else:
            # Use specified layers, validating they are Conv2d and in the pruning proposal
            valid_specified_conv_layers = [
                 l for l in layers if isinstance(self.get_layer_by_name(l), nn.Conv2d)
            ]
            if not valid_specified_conv_layers:
                 global_logger.warning(f"None of the specified layers ({layers}) are Conv2d layers. Cannot visualize gradients.")
                 return

            layers_to_visualize = [
                l for l in valid_specified_conv_layers if l in self.pruning_proposal
            ]
            if not layers_to_visualize:
                 global_logger.warning(f"None of the specified Conv2d layers ({valid_specified_conv_layers}) were found in the pruning proposal. Cannot visualize gradients.")
                 return
            global_logger.info(f"Visualizing gradients for specified Conv2d layers found in pruning proposal: {layers_to_visualize}")

        # Prepare input samples (reuse the logic from visualize_pruned_vs_kept)
        inputs_to_process: List[torch.Tensor] = []
        if use_loader:
            if isinstance(self.dataset, DataLoader):
                loader = self.dataset
            else:
                loader = DataLoader(
                    self.dataset,
                    batch_size=1, # Process one sample at a time for visualization
                    shuffle=random_sampling,
                    num_workers=max(1, os.cpu_count() // 4)
                )
            if random_sampling and isinstance(loader.dataset, torch.utils.data.Dataset):
                dataset_size = len(loader.dataset)
                if dataset_size == 0:
                     global_logger.error("Dataset is empty. Cannot sample inputs for visualization.")
                     return
                indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
                for i in indices:
                    inputs_to_process.append(loader.dataset[i][0].unsqueeze(0).to(self.device))
            else:
                global_logger.warning("Random sampling not available or dataset type not standard Dataset. Iterating through loader sequentially.")
                for inputs, _ in loader:
                    for img in inputs.split(1):
                        inputs_to_process.append(img.to(self.device))
                        if len(inputs_to_process) >= num_samples: break
                    if len(inputs_to_process) >= num_samples: break
                inputs_to_process = inputs_to_process[:num_samples]
            if not inputs_to_process:
                 global_logger.error("Could not collect any input samples from the loader.")
                 return
        else:
            if sample_inputs is None or not sample_inputs:
                raise ValueError("sample_inputs must be provided when use_loader=False and cannot be empty.")
            inputs_to_process = [inp.to(self.device) if inp.ndim == 4 else inp.unsqueeze(0).to(self.device)
                               for inp in sample_inputs[:num_samples]]


        def denorm_img(t: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
            """Helper function to denormalize a tensor image for visualization."""
            if t.ndim not in [3, 4]:
                global_logger.warning(f"Unsupported tensor dimensions for denormalization: {t.ndim}")
                return t.squeeze().cpu().numpy() if t.numel() > 0 else np.array([])
            img = t.clone().cpu().squeeze(0)
            if denormalize:
                if len(mean) != img.size(0) or len(std) != img.size(0):
                     global_logger.warning(
                         f"Mean/std channels ({len(mean)}/{len(std)}) mismatch image channels ({img.size(0)}). "
                         "Skipping denormalization."
                     )
                else:
                    for c, (m, s) in enumerate(zip(mean, std)): img[c] = img[c] * s + m
            img = img.permute(1, 2, 0).numpy()
            return np.clip(img, 0, 1)

        global_logger.info(f"Visualizing gradients for {len(inputs_to_process)} samples...")
        # Use tqdm to show progress for sample processing
        for idx, inp in tqdm(enumerate(inputs_to_process), total=len(inputs_to_process), desc='Visualizing sample gradients'):
            # Ensure input requires gradients
            inp.requires_grad_(True)
            intermediate_activations: Dict[str, torch.Tensor] = {}
            hook_handles = []

            # Define a hook to save the output tensor of the target layers
            def create_save_hook(layer_name: str):
                def hook(_, __, output: torch.Tensor):
                     # Save the output tensor reference. Gradients will be computed w.r.t this tensor later.
                     intermediate_activations[layer_name] = output
                return hook

            # Attach hooks to the target layers
            for name in layers_to_visualize:
                layer = self.get_layer_by_name(name)
                if layer is None: continue # Already warned
                if isinstance(layer, nn.Conv2d):
                     hook_handles.append(layer.register_forward_hook(create_save_hook(name)))

            # Perform the forward pass. Need to retain the graph to compute gradients later.
            self.model.eval() # Typically gradients are computed during training, but eval mode is fine here
            try:
                # Set retain_graph=True to allow backward() to be called multiple times
                # or to use autograd.grad after the forward pass.
                output_logits = self.model(inp.to(self.device))
            except Exception as e:
                global_logger.error(f"Error during forward pass for gradient visualization sample {idx}: {e}. Skipping sample.")
                for h in hook_handles: h.remove() # Clean up hooks
                continue

            # --- Compute Gradients ---
            gradients: Dict[str, torch.Tensor] = {}
            try:
                 # Identify the target scalar for backpropagation
                 if target_class_for_grad is not None:
                      if not (0 <= target_class_for_grad < output_logits.size(1)):
                           predicted_class = output_logits.argmax(dim=1).item()
                           target_score = output_logits[0, predicted_class]
                           grad_target_class_idx = predicted_class
                           global_logger.warning(f"Target class {target_class_for_grad} out of bounds. Using predicted class {predicted_class}.")
                      else:
                           target_score = output_logits[0, target_class_for_grad]
                           grad_target_class_idx = target_class_for_grad
                           global_logger.info(f"Using specified target class {target_class_for_grad} for gradient target.")
                 else:
                      predicted_class = output_logits.argmax(dim=1).item()
                      target_score = output_logits[0, predicted_class]
                      grad_target_class_idx = predicted_class
                      global_logger.info(f"Using predicted class {predicted_class} for gradient target.")

                 # Prepare inputs for autograd.grad - these are the intermediate activation tensors
                 grad_inputs = [intermediate_activations[name] for name in layers_to_visualize if name in intermediate_activations]

                 if not grad_inputs:
                      global_logger.warning(f"No intermediate activations captured for gradient computation for sample {idx}.")
                      gradients = {}
                 else:
                     # Compute gradients of the target score w.r.t. the captured intermediate activations
                     # The graph from inp to output_logits was created in the forward pass
                     # autograd.grad will traverse this graph backward from target_score to grad_inputs
                     grads_tuple = torch.autograd.grad(
                         outputs=target_score, # Scalar output to backpropagate from
                         inputs=grad_inputs,   # Tensors w.r.t which gradients are computed
                         retain_graph=True,   # Retain graph to potentially compute gradients for other inputs/outputs later (optional here, depends on flow)
                         create_graph=False    # Do not create graph for gradient computation (first-order gradients)
                     )
                     # Map the computed gradients back to the layer names
                     gradients = {name: grad.detach() for name, grad in zip([name for name in layers_to_visualize if name in intermediate_activations], grads_tuple)} # Detach gradients for plotting

            except Exception as e:
                 global_logger.error(f"Error computing gradients for sample {idx}: {e}. Skipping gradient visualization for this sample.")
                 gradients = {} # Set gradients to empty if computation fails

            finally:
                # Ensure input gradients are cleared and hooks are removed
                if inp.grad is not None:
                    inp.grad.zero_()
                for h in hook_handles: h.remove()


            # Plotting gradients
            n_layers = len(layers_to_visualize)
            fig, axes = plt.subplots(n_layers, 4, figsize=(12, 3 * n_layers))
            if n_layers == 1:
                axes = axes.reshape(1, 4)

            orig_img = denorm_img(inp.detach(), mean, std) # Detach input for plotting

            for i, name in enumerate(layers_to_visualize):
                grad_map_full = gradients.get(name) # Get the computed gradient tensor for this layer

                if grad_map_full is None:
                    global_logger.warning(f"Gradient not available for layer {name} for sample {idx}. Skipping visualization row.")
                    for col in range(4): axes[i, col].axis('off')
                    continue

                # Get pruned and kept indices (reuse from pruning proposal)
                pruned_idxs = self.pruning_proposal.get(name, [])
                # Get all possible filter indices for this layer based on the gradient map shape
                all_filter_indices = list(range(grad_map_full.size(1)))
                kept_idxs = [j for j in all_filter_indices if j not in pruned_idxs]


                # Check if we have both pruned and kept filters to select from
                if not pruned_idxs:
                     global_logger.warning(f"No pruned filters found for layer {name}. Cannot visualize pruned gradient.")
                     # Still plot kept filter and overlay if possible
                     if kept_idxs:
                          pr = None # No pruned filter to select
                          kp = random.choice(kept_idxs) # Select a random kept filter
                     else:
                          global_logger.warning(f"No kept filters found for layer {name}. Cannot visualize.")
                          for col in range(4): axes[i, col].axis('off')
                          continue
                elif not kept_idxs:
                     global_logger.warning(f"No kept filters found for layer {name}. Cannot visualize kept gradient or overlay.")
                     if pruned_idxs:
                          pr = random.choice(pruned_idxs) # Select a random pruned filter
                          kp = None # No kept filter to select
                     else:
                          global_logger.warning(f"No pruned or kept filters for layer {name}. Cannot visualize.")
                          for col in range(4): axes[i, col].axis('off')
                          continue
                else:
                    # Randomly select one pruned and one kept filter index
                    pr = random.choice(pruned_idxs)
                    kp = random.choice(kept_idxs)


                # Column 1: Original image
                axes[i, 0].imshow(orig_img)
                axes[i, 0].set_title(f"Sample {idx}\nOriginal")
                axes[i, 0].axis('off')

                # Column 2: Pruned filter gradient map
                if pr is not None:
                    grad_pruned_filter = grad_map_full[0, pr] # Get gradient for the selected pruned filter (batch 0)
                    vis_grad_pruned = torch.abs(grad_pruned_filter).cpu().numpy()
                    axes[i, 1].imshow(vis_grad_pruned, cmap='hot')
                    axes[i, 1].set_title(f"{name}\nPruned Grad #{pr}")
                else:
                     axes[i, 1].text(0.5, 0.5, 'No Pruned\nFilter', horizontalalignment='center', verticalalignment='center', transform=axes[i, 1].transAxes)
                axes[i, 1].axis('off')

                # Column 3: Kept filter gradient map
                if kp is not None:
                    grad_kept_filter = grad_map_full[0, kp] # Get gradient for the selected kept filter (batch 0)
                    vis_grad_kept = torch.abs(grad_kept_filter).cpu().numpy()
                    axes[i, 2].imshow(vis_grad_kept, cmap='hot')
                    axes[i, 2].set_title(f"{name}\nKept Grad #{kp}")
                else:
                     axes[i, 2].text(0.5, 0.5, 'No Kept\nFilter', horizontalalignment='center', verticalalignment='center', transform=axes[i, 2].transAxes)
                axes[i, 2].axis('off')

                # Column 4: Heatmap overlay of kept filter gradient on input
                if kp is not None:
                    axes[i, 3].imshow(orig_img) # Background is the original image
                    heatmap = vis_grad_kept # Use the absolute gradient for heatmap
                    if heatmap.max() == heatmap.min():
                        heatmap_norm = np.zeros_like(heatmap)
                    else:
                         heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6) # Add epsilon
                    axes[i, 3].imshow(heatmap_norm, cmap='hot', alpha=0.6)
                    axes[i, 3].set_title(f"{name}\nGrad Overlay #{kp}")
                else:
                     axes[i, 3].text(0.5, 0.5, 'No Kept\nOverlay', horizontalalignment='center', verticalalignment='center', transform=axes[i, 3].transAxes)
                axes[i, 3].axis('off')


            plt.tight_layout()

            # Save figure if buffering is enabled
            if buffer:
                try:
                    os.makedirs(buffer_path, exist_ok=True)
                    suffix = uuid.uuid4().hex[:8]
                    # Include predicted/target class in filename
                    fname = f"{buffer_prefix}_sample{idx}_class{grad_target_class_idx}_{suffix}.png"
                    path = os.path.join(buffer_path, fname)
                    fig.savefig(path)
                    global_logger.info(f"Saved gradient visualization to {path}")
                except Exception as e:
                    global_logger.warning(f"Failed to save gradient buffer for sample {idx}: {e}")

            # Display figure unless only saving is requested
            if not just_save_data:
               plt.show()
            else:
               plt.close(fig)

    def visualize_pruned_vs_kept(
        self,
        layers: Optional[List[str]] = None,
        use_loader: bool = True, # Changed default to True
        num_samples: int = 3, # Increased default samples
        random_sampling: bool = True, # Changed default to True
        sample_inputs: Optional[List[torch.Tensor]] = None,
        denormalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406], # Default ImageNet mean
        std: List[float] = [0.229, 0.224, 0.225], # Default ImageNet std
        buffer: bool = False,
        buffer_path: str = './visualization_buffers', # Changed default path
        buffer_prefix: str = 'vis',
        just_save_data: bool = False, # Added flag to only save, not display
        least_and_most: bool = False, # Added flag to visualize least and most activated filters
        least_and_most_samples: int = 1, # Number of least and most activated filters to visualize
        after_pruning: bool = False # Added flag to visualize after pruning
    ) -> None:
        """
        Visualizes the impact of pruning by displaying:
        1. The original input image.
        2. The activation map of a *pruned* filter in a specified layer.
        3. The activation map of a *kept* filter in the same layer.
        4. A heatmap overlay of the *kept* filter's activation on the input image.

        This helps understand what features the pruned filters were responding to
        compared to the features captured by the kept filters.

        Args:
            layers: A list of specific layer names (strings) to visualize.
                    Defaults to all layers that were pruned according to `self.pruning_proposal`.
            use_loader: If True, draw input samples from `self.dataset` DataLoader.
                        If False, `sample_inputs` must be provided.
            num_samples: The number of input samples to visualize.
            random_sampling: If `use_loader` is True, randomly sample `num_samples`
                             from the dataset. If False, take the first `num_samples`.
            sample_inputs: A list of input tensors to use for visualization when
                           `use_loader` is False.
            denormalize: If True, attempt to reverse the normalization using `mean`
                         and `std` to display a standard image.
            mean: List of mean values used for normalization (for denormalization).
            std: List of standard deviation values used for normalization (for denormalization).
            buffer: If True, save the generated visualization figures to disk.
            buffer_path: Directory to save the visualization images if `buffer` is True.
                         Created if it does not exist.
            buffer_prefix: A string prefix for the filenames of saved images.
            just_save_data: If True, save the figures to `buffer_path` but do not display them using `plt.show()`.
        """
        if self.pruning_proposal is None or not self.pruning_proposal:
            global_logger.warning("No pruning proposal found. Run prune_online() first.")
            return

        # Select layers to visualize
        if layers is None:
            # Default to all layers that were pruned
            layers_to_visualize = list(self.pruning_proposal.keys())
            if not layers_to_visualize:
                global_logger.warning("No layers were pruned according to the pruning proposal. Cannot visualize.")
                return
            global_logger.info(f"Visualizing all pruned layers: {layers_to_visualize}")
        else:
            layers_to_visualize = layers
            # Validate that the specified layers were actually pruned
            valid_layers = [l for l in layers_to_visualize if l in self.pruning_proposal and self.pruning_proposal[l]]
            if not valid_layers:
                 global_logger.warning(f"None of the specified layers ({layers}) were pruned or found in the pruning proposal. Cannot visualize.")
                 return
            layers_to_visualize = valid_layers
            global_logger.info(f"Visualizing specified pruned layers: {layers_to_visualize}")


        # Prepare input samples
        inputs_to_process: List[torch.Tensor] = []
        if use_loader:
            if isinstance(self.dataset, DataLoader):
                loader = self.dataset
            else:
                # Create a DataLoader from the dataset if not already one
                loader = DataLoader(
                    self.dataset,
                    batch_size=1, # Visualize one sample at a time
                    shuffle=random_sampling, # Shuffle if random sampling is requested
                    num_workers=max(1, os.cpu_count() // 4) # Use some workers
                )

            if random_sampling and isinstance(loader.dataset, torch.utils.data.Dataset):
                 # If random sampling and dataset is a standard Dataset, sample indices directly
                dataset_size = len(loader.dataset)
                if dataset_size == 0:
                     global_logger.error("Dataset is empty. Cannot sample inputs for visualization.")
                     return
                indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
                for i in indices:
                    # Get sample from dataset, add batch dimension, move to device
                    inputs_to_process.append(loader.dataset[i][0].unsqueeze(0).to(self.device))
            else:
                # If not random sampling or dataset is not standard Dataset (e.g., a custom iterable)
                # Iterate through the loader sequentially
                global_logger.warning("Random sampling not available or dataset type not standard Dataset. Iterating through loader sequentially.")
                for inputs, _ in loader:
                    for img in inputs.split(1): # Process each image in the batch
                        inputs_to_process.append(img.to(self.device))
                        if len(inputs_to_process) >= num_samples:
                            break
                    if len(inputs_to_process) >= num_samples:
                        break
                # Limit to num_samples if more were collected (shouldn't happen with batch_size=1 logic above)
                inputs_to_process = inputs_to_process[:num_samples]

            if not inputs_to_process:
                 global_logger.error("Could not collect any input samples from the loader.")
                 return

        else: # Use provided sample_inputs
            if sample_inputs is None or not sample_inputs:
                raise ValueError("sample_inputs must be provided when use_loader=False and cannot be empty.")
            # Use the provided inputs, move to device and add batch dim if needed
            inputs_to_process = [inp.to(self.device) if inp.ndim == 4 else inp.unsqueeze(0).to(self.device)
                               for inp in sample_inputs[:num_samples]]


        def denorm_img(t: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
            """Helper function to denormalize a tensor image for visualization."""
            # Ensure input tensor is 3D (C, H, W) or 4D (B, C, H, W)
            if t.ndim not in [3, 4]:
                global_logger.warning(f"Unsupported tensor dimensions for denormalization: {t.ndim}")
                return t.squeeze().cpu().numpy() # Attempt to convert to numpy anyway

            # If it's a batch (4D), take the first image
            img = t.clone().cpu().squeeze(0) # Remove batch dim if present

            if denormalize:
                # Ensure mean and std have correct number of channels
                if len(mean) != img.size(0) or len(std) != img.size(0):
                     global_logger.warning(
                         f"Mean/std channels ({len(mean)}/{len(std)}) mismatch image channels ({img.size(0)}). "
                         "Skipping denormalization."
                     )
                else:
                    # Apply inverse normalization per channel
                    for c, (m, s) in enumerate(zip(mean, std)):
                        img[c] = img[c] * s + m

            # Permute dimensions from (C, H, W) to (H, W, C) for matplotlib
            img = img.permute(1, 2, 0).numpy()

            # Clip values to [0, 1] range for proper display
            return np.clip(img, 0, 1)

        # Visualize each sample
        global_logger.info(f"Visualizing {len(inputs_to_process)} samples...")
        # Use tqdm to show progress for sample processing
        for idx, inp in tqdm(enumerate(inputs_to_process), total=len(inputs_to_process), desc='Handling samples for visualization'):
            activations: Dict[str, torch.Tensor] = {} # Store activations for relevant layers
            handles = [] # Store hook handles for cleanup

            # Register forward hooks on the selected layers to capture activations
            for name in layers_to_visualize:
                layer = self.get_layer_by_name(name)
                if layer is None:
                    global_logger.warning(f"Layer '{name}' not found for visualization. Skipping.")
                    continue
                # Define a hook function that captures the output and stores it
                def make_hook(n=name): # Use default argument to capture layer name
                    def hook(_, __, out):
                         # Store activation for the first image in the batch (since batch size is 1)
                         activations[n] = out.detach().cpu().squeeze(0)
                    return hook
                # Register the hook and save the handle
                handles.append(layer.register_forward_hook(make_hook()))

            # Perform forward pass with the input sample to trigger the hooks
            self.model.eval() # Ensure model is in evaluation mode
            with torch.no_grad(): # Disable gradient calculation
                try:
                    _ = self.model(inp.to(self.device))
                except Exception as e:
                    global_logger.error(f"Error during forward pass for visualization sample {idx}: {e}. Skipping.")
                    # Clean up hooks before continuing to the next sample
                    for h in handles:
                        h.remove()
                    continue


            # Plotting
            n_layers = len(layers_to_visualize)
            # Create subplots: n_layers rows, 4 columns (Original, Pruned Act, Kept Act, Overlay)
            if (not after_pruning):
                fig, axes = plt.subplots(n_layers, 4, figsize=(12, 3 * n_layers))
                # If only one layer, axes will be a 1D array, reshape to 2D for consistent indexing
                if n_layers == 1:
                    axes = axes.reshape(1, 4)
                    
            else:
                fig, axes = plt.subplots(n_layers, 3, figsize=(12, 3 * n_layers))
                # If only one layer, axes will be a 1D array, reshape to 2D for consistent indexing
                if n_layers == 1:
                    axes = axes.reshape(1, 3)

            # Denormalize the input image for display
            orig_img = denorm_img(inp, mean, std)

            # Iterate through the selected layers and plot visualizations
            for i, name in enumerate(layers_to_visualize):
                act = activations.get(name) # Get the captured activation for this layer
                if act is None:
                    global_logger.warning(f"No activation captured for layer {name}. Skipping visualization for this layer.")
                    # Hide the row of subplots for this layer if no activation
                    if (not after_pruning):
                        for col in range(4):
                            axes[i, col].axis('off')
                        continue
                    else:
                        for col in range(3):
                            axes[i, col].axis('off')
                        continue

                if after_pruning:

                    pruned_idxs = [j for j in range(act.size(0))]
                    kept_idxs = [j for j in range(act.size(0))]

                else:

                    if (not least_and_most):

                        # Get the indices of pruned and kept filters for this layer
                        pruned_idxs = self.pruning_proposal.get(name, [])
                        # Find kept indices by excluding pruned indices from all indices
                        kept_idxs = [j for j in range(act.size(0)) if j not in pruned_idxs]
                        
                    else:
                        if (least_and_most_samples ==1):
                            pruned_idxs = [self.pruning_proposal[name][0]]        
                            kept_idxs   = [self.pruning_proposal[name][-1]]     
                        
                        else:
                            pruned_idxs = self.pruning_proposal[name][:least_and_most_samples]        
                            kept_idxs   = self.pruning_proposal[name][-least_and_most_samples:]   
                    
                # Check if there are both pruned and kept filters to visualize
                if not pruned_idxs:
                     global_logger.warning(f"No pruned filters found for layer {name}. Cannot visualize pruned activation.")
                     # Still plot kept filter and overlay if possible
                     if kept_idxs:
                          # Original image
                          axes[i, 0].imshow(orig_img)
                          axes[i, 0].set_title(f"Sample {idx}\nOriginal")
                          axes[i, 0].axis('off')
                           # Pruned activation (empty plot or placeholder)
                          axes[i, 1].text(0.5, 0.5, 'No Pruned', horizontalalignment='center', verticalalignment='center', transform=axes[i, 1].transAxes)
                          axes[i, 1].axis('off')
                           # Kept activation
                          kp = kept_idxs[0] # Take the first kept filter
                          axes[i, 2].imshow(act[kp], cmap='viridis')
                          axes[i, 2].set_title(f"{name}\nKept Act #{kp}")
                          axes[i, 2].axis('off')
                           # Overlay heatmap
                          axes[i, 3].imshow(orig_img)
                          heatmap = act[kp]
                           # Normalize heatmap for alpha blending
                          heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                          axes[i, 3].imshow(heatmap_norm, cmap='hot', alpha=0.6)
                          axes[i, 3].set_title(f"{name}\nOverlay #{kp}")
                          axes[i, 3].axis('off')
                     else:
                          global_logger.warning(f"No kept filters found for layer {name}. Cannot visualize.")
                          # Hide the row of subplots
                          for col in range(4):
                               axes[i, col].axis('off')
                     continue

                if not kept_idxs:
                    global_logger.warning(f"No kept filters found for layer {name}. Cannot visualize kept activation or overlay.")
                    # Still plot original and pruned activation if possible
                    if pruned_idxs:
                         # Original image
                         axes[i, 0].imshow(orig_img)
                         axes[i, 0].set_title(f"Sample {idx}\nOriginal")
                         axes[i, 0].axis('off')
                          # Pruned activation
                         pr = pruned_idxs[0] # Take the first pruned filter
                         axes[i, 1].imshow(act[pr], cmap='viridis')
                         axes[i, 1].set_title(f"{name}\nPruned Act #{pr}")
                         axes[i, 1].axis('off')
                         # Kept activation (empty plot or placeholder)
                         axes[i, 2].text(0.5, 0.5, 'No Kept', horizontalalignment='center', verticalalignment='center', transform=axes[i, 2].transAxes)
                         axes[i, 2].axis('off')
                         # Overlay heatmap (empty plot or placeholder)
                         axes[i, 3].text(0.5, 0.5, 'No Kept', horizontalalignment='center', verticalalignment='center', transform=axes[i, 3].transAxes)
                         axes[i, 3].axis('off')
                    else:
                         global_logger.warning(f"No pruned or kept filters for layer {name}. Cannot visualize.")
                         # Hide the row of subplots
                         for col in range(4):
                              axes[i, col].axis('off')
                    continue

                # If we have both pruned and kept filters, proceed with full visualization
                pr = pruned_idxs.copy()
                kp = kept_idxs.copy()

                pr_act = torch.max(act[pr], dim=0)[0]
                kp_act = torch.max(act[kp], dim=0)[0]
                
                max_pr_act = torch.max(pr_act)
                max_kp_act = torch.max(kp_act)
                
                min_pr_act = torch.min(pr_act)
                min_kp_act = torch.min(kp_act)
                
                overall_min = min(min_pr_act, min_kp_act)
                overall_max = max(max_pr_act, max_kp_act)

                if (after_pruning):

                    # Column 1: Original image
                    axes[i, 0].imshow(orig_img)
                    # Add sample index and "Original" title
                    axes[i, 0].set_title(f"Sample {idx}\nOriginal")
                    axes[i, 0].axis('off') # Hide axes ticks and labels

                    # Column 2: Pruned filter activation
                    # Display the activation map for the selected pruned filter
                    axes[i, 1].imshow(pr_act, cmap='viridis', vmin=overall_min, vmax=overall_max) # Use viridis colormap
                    # Add layer name and filter index to title
                    if (not least_and_most):
                        axes[i, 1].set_title(f"{name}\nActivation #{len(pr)}")
                    else:
                        axes[i, 1].set_title(f"{name}\nnActivation #{len(pr)}")
                    axes[i, 1].axis('off')
                    
                    # Column 4: Heatmap overlay of kept filter activation on the input image
                    axes[i, 2].imshow(orig_img) # Display the original image as background
                    heatmap = kp_act # Get the activation map of the kept filter
                    # Normalize the heatmap to [0, 1] for alpha blending
                    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6) # Add epsilon to avoid division by zero
                    # Overlay the heatmap with transparency
                    axes[i, 2].imshow(heatmap_norm, cmap='hot', alpha=0.6) # Use hot colormap with transparency
                    # Add layer name and filter index to title
                    axes[i, 2].set_title(f"{name}\nOverlay #{kp}")
                    axes[i, 2].axis('off')
                    
                else:
                    
                                        # Column 1: Original image
                    axes[i, 0].imshow(orig_img)
                    # Add sample index and "Original" title
                    axes[i, 0].set_title(f"Sample {idx}\nOriginal")
                    axes[i, 0].axis('off') # Hide axes ticks and labels

                    # Column 2: Pruned filter activation
                    # Display the activation map for the selected pruned filter
                    axes[i, 1].imshow(pr_act, cmap='viridis', vmin=overall_min, vmax=overall_max) # Use viridis colormap
                    # Add layer name and filter index to title
                    if (not least_and_most):
                        axes[i, 1].set_title(f"{name}\nPruned Act #{len(pr)}")
                    else:
                        axes[i, 1].set_title(f"{name}\nMost Act #{len(pr)}")
                    axes[i, 1].axis('off')

                    # Column 3: Kept filter activation
                    # Display the activation map for the selected kept filter
                    axes[i, 2].imshow(kp_act, cmap='viridis', vmin=overall_min, vmax=overall_max) # Use viridis colormap
                    # Add layer name and filter index to title
                    if (not least_and_most):
                        axes[i, 2].set_title(f"{name}\nKept Act #{len(kp)}")
                    else:
                        axes[i, 2].set_title(f"{name}\nLeast Act #{len(kp)}")
                    axes[i, 2].axis('off')

                    # Column 4: Heatmap overlay of kept filter activation on the input image
                    axes[i, 3].imshow(orig_img) # Display the original image as background
                    heatmap = kp_act # Get the activation map of the kept filter
                    # Normalize the heatmap to [0, 1] for alpha blending
                    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6) # Add epsilon to avoid division by zero
                    # Overlay the heatmap with transparency
                    axes[i, 3].imshow(heatmap_norm, cmap='hot', alpha=0.6) # Use hot colormap with transparency
                    # Add layer name and filter index to title
                    axes[i, 3].set_title(f"{name}\nOverlay #{kp}")
                    axes[i, 3].axis('off')
                    
                    

            # Adjust layout to prevent titles/labels overlapping
            plt.tight_layout()

            # Save figure if buffering is enabled
            if buffer:
                try:
                    # Create buffer directory if it doesn't exist
                    os.makedirs(buffer_path, exist_ok=True)
                    fname = f"{buffer_prefix}_sample{idx}.png"
                    path = os.path.join(buffer_path, fname)
                    fig.savefig(path)
                except Exception as e:
                    global_logger.warning(f"Failed to save buffer for sample {idx}: {e}")

            # Display figure unless only saving is requested
            if not just_save_data:
                plt.show()
            else:
                # Close the figure to free memory if not displaying
                plt.close(fig)

            # Clean up hooks after processing each sample
            for h in handles:
                h.remove()

    def get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """
        Retrieves a specific submodule (layer) from the model by its full name
        as returned by `model.named_modules()`.

        Args:
            name: The full name of the module (e.g., 'features.0', 'classifier.fc1').

        Returns:
            The nn.Module instance if found, otherwise None.
        """
        # Get a dictionary mapping all module names to their instances
        named_modules = dict(self.model.named_modules())
        # Return the module if the name exists in the dictionary, otherwise None
        return named_modules.get(name, None)

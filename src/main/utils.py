import torch
import torch
import torch.nn as nn

def export_data(trainer, pruned_model_path, base_pruned_model_paths, dataset,
    model_name, dataset_name, pruning_rate):
    """
    Export the pruned model and its state dict to a specified path.
    Also saves the model as JIT for both CPU and GPU.
    """
    #Saving the pruned model and its state dictionary
    torch.save(trainer.model, f'{pruned_model_path}.pt')
    torch.save(trainer.model.state_dict(), f'{pruned_model_path}_state_dict.pt')

     # Saving the model as JIT - cpu
    trainer.model.to('cpu')
    traced_model = torch.jit.trace(trainer.model, torch.randn(1, 3, 224, 224).to('cpu'))
    traced_model.save(f'{pruned_model_path}_cpu.jit')
    
    # Saving the model as JIT - GPU
    trainer.model.to('cuda')
    traced_model = torch.jit.trace(trainer.model, torch.randn(1, 3, 224, 224).to('cuda'))
    traced_model.save(f'{pruned_model_path}_gpu.jit')
           
    # Evaluate the pruned model
    trainer.evaluate_model(
         dataset,
         generate_log=True,
         path_log=f"{base_pruned_model_paths}",
         prefix=f"benchmark_val_{model_name}_{dataset_name}_{pruning_rate}_"
         
     )

def is_output_shape_valid(output: torch.Tensor) -> bool:
    """
    Check that no dimension of the output tensor is zero.

    Args:
        output: a torch.Tensor produced by your layer.

    Returns:
        True if all dimensions are > 0, False if any dimension == 0.
    """
    return all(dim > 0 for dim in output.shape)


def get_last_layer_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Run a forward pass through `model` on input `x` and return the output
    of its last layer, captured via a forward hook.

    Args:
        model: any torch.nn.Module
        x: input tensor to feed into the model

    Returns:
        The output tensor from the last layer.
    """
    activations = {}

    def grab_last_hook(module, inp, outp):
        # store the last-layer output
        activations['last'] = outp

    # Identify the last submodule in the model
    last_module = list(model.modules())[-1]
    # Register the hook
    handle = last_module.register_forward_hook(grab_last_hook)

    flg_valid_model = False

    # Forward pass
    model.eval()
    
    try:
    
        with torch.no_grad():
            _ = model(x)
            
        flg_valid_model = True
    except Exception as e:
        print(f"Error during forward pass: {e}")
        
        flg_valid_model = False

    # Remove hook
    handle.remove()

    return flg_valid_model

    
    
def export_viz_pre_pruning(gap_pruning, model_name, dataset_name, pruning_rate,
                          sample_viz, buffer_path, list_of_layers_to_visualize, pruned_data,
                          full_pruned_data):
    """
    Export the visualization of the pruned model before pruning.
    """
    gap_pruning.pruning_proposal = pruned_data
    gap_pruning.visualize_pruned_vs_kept(use_loader=True,
                 num_samples=sample_viz,
                 random_sampling = True,
                 layers=list_of_layers_to_visualize,
                 buffer= True,
                 buffer_path= buffer_path,
                 buffer_prefix=f"viz_layerwise_{model_name}_{dataset_name}_{pruning_rate}",
                 just_save_data= True,
                 least_and_most = False,)

    # Just for the max and least pruned layers
    gap_pruning.pruning_proposal = full_pruned_data
    gap_pruning.visualize_pruned_vs_kept(use_loader=True,
                 num_samples=sample_viz,
                 random_sampling = True,
                 layers=list_of_layers_to_visualize,
                 buffer= True,
                 buffer_path= buffer_path,
                 buffer_prefix=f"viz_channel_{model_name}_{dataset_name}_{pruning_rate}",
                 just_save_data= True,
                 least_and_most=True,
                 least_and_most_samples=5)
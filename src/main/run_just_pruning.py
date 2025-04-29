import torch.optim as optim
import torch.nn as nn
from gap_pruning_interactive import GapPruningInteractive as GapPruning
from train_pruned import TrainPrunedModel
from train import TrainModel
import os
import torch
import warnings
from rich.console import Console
import copy
import logging
from utils import *

warnings.filterwarnings("ignore")

# Configure module-level logger
global_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


console = Console()

if __name__ == '__main__':

    actual_path = os.getcwd()
    console.print(f"[bold blue]Running on:[/bold blue] [white]{actual_path}[/white]")

    batch_size = 16

    models_arr   = [
    "resnet50.ra_in1k"
    ]

    dataset_arr  = ['flowers102', 'food101']
    classes_arr  = [102,101]
    classes_arr  = [100, 102, 101]
    pruning_rates = [.05]#,.1,.15,.2,.25,.3,.35]

    dict_list_of_layers_to_visualize = {"resnet50.ra_in1k":['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1']}
    epochs = 1
    sample_viz = 5
    skip_initial_eval = False
    base_model_paths = f'{actual_path}/checkpoints/base_trained_models'
    base_pruned_model_paths = f'{actual_path}/checkpoints/base_prunned_models'
    buffer_path = f'{actual_path}/buffers'

    for model_name in models_arr:
        for dataset_name, num_classes in zip(dataset_arr, classes_arr):
            
            # Setting dummy trainer with base model
            dummy_train = TrainModel()
            dummy_train.set_model(id_model=model_name, num_classes=num_classes)
            dummy_train.set_dataset(id_dataset=dataset_name, batch_size=batch_size, download=True)

            console.rule(
                f"[bold magenta] Model:[/bold magenta] [white]{model_name}[/white]  •  "
                f"[bold magenta]Dataset:[/bold magenta] [white]{dataset_name}[/white]"
            )
            
            # Gettin the weights or the model
            model_path = f'{base_model_paths}/{model_name}/loss_full_model_{model_name}_{dataset_name}.pt'
            
            try:
                
                dummy_train.load_weights(model_path)
                console.rule(f"[bold magenta] Model Weights Loaded:[/bold magenta] [white]{model_name}[/white]  •  ")
                
            except:
                
                dummy_train.model = torch.load(model_path, weights_only=False)
                console.rule(f"[bold magenta] Full Model Loaded:[/bold magenta] [white]{model_name}[/white]  •  ")
            
            # Evaluate the original model
            optimizer = optim.SGD(dummy_train.model.parameters(),lr=0.001, momentum=0.9)
            dummy_train.set_optimizer(optimizer)
            dummy_train.set_criterion(nn.CrossEntropyLoss())

            scheduler = optim.lr_scheduler.MultiStepLR(dummy_train.optimizer, milestones=[75, 150], gamma=0.5)
            dummy_train.set_scheduler(scheduler)

            if (not skip_initial_eval):
                dummy_train.evaluate_model(
                            dummy_train.arr_dataset[1],
                            generate_log=True,
                            path_log=f"{base_pruned_model_paths}",
                            prefix=f"benchmark_{model_name}_{dataset_name}_"
                        )
            
            console.print("[magenta]▸ Starting GapPruning…[/magenta]")
            parameters_before = dummy_train.count_parameters()
            
            base_model = copy.deepcopy(dummy_train.model)
            
            for pruning_rate in pruning_rates:

                gap_pruning = GapPruning(model=copy.deepcopy(base_model), dataset=dummy_train.arr_dataset[0], device='cuda')
                gap_pruning.model = gap_pruning.model.to('cuda')
                
                # Compute the activations
                global_logger.info(f"->Starting computing metrics!")
                gap_pruning.compute_stats_on_the_fly(sample_limit=1000, 
                                                     load=True,
                                                     file_path=f"stats_{model_name}_{dataset_name}.pt",)

                # Pruning process
                pruned_report, pruned_data, full_pruned_data = gap_pruning.prune_online(global_prune_rate=pruning_rate)
                
                # Forcing the hooks to be removed
                gap_pruning._remove_hooks()
                
                # Lets check if is a valid model
                is_valid_rate  = get_last_layer_output(gap_pruning.model, torch.randn(1, 3, 224, 224).to('cuda'))

                if not is_valid_rate:
                    console.print("[red]▸ Pruning rate is not valid, skipping the rest of the rates![/red]")
                    break

                # Checking the proposed pruned layers
                pruned_model      = copy.deepcopy(gap_pruning.model)
                gap_pruning.model = copy.deepcopy(base_model)

                # Generating images for the prune and not pruned layer                
                list_of_layers_to_visualize = dict_list_of_layers_to_visualize[model_name]
                
                # Generate images for the prune and not pruned layer - Before pruning
                export_viz_pre_pruning(gap_pruning, model_name, dataset_name, pruning_rate,
                                       sample_viz, buffer_path, list_of_layers_to_visualize, pruned_data,
                                       full_pruned_data)
                
                # Finetune pruned model
                trainer = TrainPrunedModel(is_gpu=True)
                
                # Setting parameters
                trainer.model = copy.deepcopy(pruned_model)
                trainer.optimizer = optim.SGD(trainer.model.parameters(),lr=0.001, momentum=0.9)
                trainer.criterion = nn.CrossEntropyLoss()
                trainer.scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer, milestones=[75, 150], gamma=0.5)

                # Run fine-tuning
                history = trainer.train_pruned(
                    train_loader=dummy_train.arr_dataset[0],
                    val_loader=None,
                    epochs=epochs,
                    freeze_conv=True,
                    unfreeze_after=int(epochs/10),
                    differential_lr=True,
                    lr_head=1e-2,
                    lr_conv=1e-3,
                    weight_decay=1e-4,
                    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                        trainer.optimizer, T_max=10
                    ),
                    unfreeze_scheduler_step=True,
                    pct_val=0.1,
                    generate_log=True,
                    path_log=f"{base_pruned_model_paths}",
                    prefix=f"benchmark_val_{model_name}_{dataset_name}_{pruning_rate}_"
                )                  
                
                # Save the pruned model
                pruned_model_path = f'{base_pruned_model_paths}/gap_pruned_model_{model_name}_{dataset_name}_{pruning_rate}'
                
                # Saving the pruned model and its state dictionary)
                trainer.model.load_state_dict(torch.load('best_val.pth', weights_only=True))
                
                #Salvando os dados de treino
                export_data(trainer, pruned_model_path, base_pruned_model_paths, dummy_train.arr_dataset[1],
                             model_name, dataset_name, pruning_rate)

                        
                # Generating images for the prune and not pruned layer - After pruning
                gap_pruning.model = trainer.model

                # Just for the max and least pruned layers
                gap_pruning.pruning_proposal = full_pruned_data
                
                gap_pruning.visualize_pruned_vs_kept(
                                                     num_samples=sample_viz,
                                                     layers=list_of_layers_to_visualize,
                                                     buffer= True,
                                                     buffer_path= buffer_path,
                                                     buffer_prefix=f"tunned_viz_channel_{model_name}_{dataset_name}_{pruning_rate}",
                                                     just_save_data= True,
                                                     least_and_most=True,
                                                     least_and_most_samples=sample_viz,
                                                     after_pruning=True)

                    
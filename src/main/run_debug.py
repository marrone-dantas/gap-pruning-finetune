from train import TrainModel
import torch.optim as optim
import torch.nn as nn
from gap_pruning import GapPruning
import os
import torch
from torchinfo import summary
import warnings
from telegram_handler import *
from rich.console import Console

warnings.filterwarnings("ignore")

console = Console()

if __name__ == '__main__':

    actual_path = os.getcwd()
    console.print(f"[bold blue]Running on:[/bold blue] [white]{actual_path}[/white]")

    batch_size = 2

    models_arr   = [
    "vgg16.tv_in1k",
    "resnet50.ra_in1k",
    "efficientnet_b0.ra_in1k",
    "mobilenetv2_100.ra_in1k",
    "convnext_base.fb_in1k",
    "densenet121.ra_in1k",
    "regnety_032.ra_in1k",
    "repvgg_a0.rvgg_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "mixnet_s.ft_in1k"
    ]
    dataset_arr  = ['cifar10','cifar100', 'flowers102', 'food101']
    classes_arr  = [10,100, 102, 101]
    pruning_rates = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    epochs = 250

    for model_name in models_arr:
        for dataset_name, num_classes in zip(dataset_arr, classes_arr):
            
            console.rule(
                f"[bold magenta] Model:[/bold magenta] [white]{model_name}[/white]  •  "
                f"[bold magenta]Dataset:[/bold magenta] [white]{dataset_name}[/white]"
            )
            for pruning_rate in pruning_rates:
                
                try:
                    skip_initial = (pruning_rate != 5)
                    console.print(f"[bold yellow]→ Pruning rate:[/bold yellow] [cyan]{pruning_rate}%[/cyan]")

                    # 1) Initial setup
                    train_handler = TrainModel()
                    train_handler.set_model(id_model=model_name, num_classes=num_classes)
                    train_handler.set_dataset(id_dataset=dataset_name, batch_size=batch_size, download=True)

                    optimizer = optim.SGD(train_handler.model.parameters(), lr=0.001, momentum=0.9)
                    train_handler.set_optimizer(optimizer)
                    train_handler.set_criterion(nn.CrossEntropyLoss())

                    scheduler = optim.lr_scheduler.MultiStepLR(train_handler.optimizer, milestones=[75, 150], gamma=0.5)
                    train_handler.set_scheduler(scheduler)

                    # 2) Initial training or load existing
                    if not skip_initial:
                        console.print("[green]▸ Starting initial training…[/green]")
                        with console.status(
                            f"[green]Training {model_name}_{dataset_name} ({epochs} epochs)…[/green]",
                            spinner="dots"
                        ):
                            train_handler.train_model(
                                epochs=epochs,
                                sufix=f"{model_name}_{dataset_name}",
                                weight_path=f"{actual_path}/checkpoints",
				                patience=25
                            )
                        console.print("[green]✔ Initial training complete![/green]")

                        train_handler.save_hist(
                            f"{actual_path}/backlog/{model_name}_{dataset_name}_output.csv"
                        )
                        train_handler.load_weights(
                            f"{actual_path}/checkpoints/loss_weights_{model_name}_{dataset_name}"
                        )
                        train_handler.evaluate_model(
                            train_handler.arr_dataset[1],
                            generate_log=True,
                            path_log=f"{actual_path}/backlog",
                            prefix=f"eval_{model_name}_{dataset_name}_"
                        )
                    else:
                        console.print("[cyan]▸ Loading existing weights…[/cyan]")
                        train_handler.load_weights(
                            f"{actual_path}/checkpoints/loss_weights_{model_name}_{dataset_name}"
                        )
                        train_handler.evaluate_model(
                            train_handler.arr_dataset[1],
                            generate_log=True,
                            path_log=f"{actual_path}/backlog",
                            prefix=f"eval_{model_name}_{dataset_name}_"
                        )

                    # 3) Model summary
                    console.print("[blue]▸ Generating model summary…[/blue]")
                    trained_model = train_handler.model.to('cuda')
                    summary(trained_model, input_size=(1, 3, 224, 224), verbose=1)

                    # 4) Gap Pruning
                    console.print("[magenta]▸ Starting GapPruning…[/magenta]")
                    gap_pruning = GapPruning(model=trained_model, dataset=train_handler.arr_dataset[0], device='cuda')
                    gap_pruning.process_dataset()
                    gap_pruning.compute_std_devs()
                    gap_pruning.generate_pruning_proposal(pruning_percentage=pruning_rate)
                    gap_pruning.prune()
                    torch.save(
                        gap_pruning.model.state_dict(),
                        f"{actual_path}/checkpoints/{model_name}_{dataset_name}_{pruning_rate}.pth"
                    )
                    gap_pruning.remove_hooks()

                    # 5) Fine-tuning pruned model
                    console.print("[yellow]▸ Fine-tuning pruned model…[/yellow]")
                    train_handler_prune = TrainModel()
                    train_handler_prune.model = gap_pruning.model
                    train_handler_prune.set_dataset(id_dataset=dataset_name, batch_size=batch_size, download=True)

                    optimizer = optim.SGD(train_handler_prune.model.parameters(), lr=0.001, momentum=0.9)
                    train_handler_prune.set_optimizer(optimizer)
                    train_handler_prune.set_criterion(nn.CrossEntropyLoss())
                    train_handler_prune.set_scheduler(
                        optim.lr_scheduler.MultiStepLR(train_handler_prune.optimizer, milestones=[75,150], gamma=0.5)
                    )

                    with console.status(
                        f"[yellow]Fine-tuning pruned_{pruning_rate}_{model_name}_{dataset_name} ({epochs} epochs)…[/yellow]",
                        spinner="dots"
                    ):
                        train_handler_prune.train_model(
                            epochs=epochs,
                            sufix=f"pruned_{pruning_rate}_{model_name}_{dataset_name}",
                            weight_path=f"{actual_path}/checkpoints",
			    patience=25,
                                   telegram_token = telegram_token,
                            telegram_chat_id = telegram_chat_id
                        )
                    console.print("[green]✔ Fine-tuning complete![/green]")

                    train_handler_prune.save_hist(
                        f"{actual_path}/backlog/pruned_{pruning_rate}_{model_name}_{dataset_name}_output.csv"
                    )
                    train_handler_prune.load_weights(
                        f"{actual_path}/checkpoints/loss_weights_pruned_{pruning_rate}_{model_name}_{dataset_name}"
                    )
                    train_handler_prune.evaluate_model(
                        train_handler_prune.arr_dataset[1],
                        generate_log=True,
                        path_log=f"{actual_path}/backlog",
                        prefix=f"pruned_{pruning_rate}_eval_{model_name}_{dataset_name}_"
                    )

                    console.print()  # blank line between iterations
                    
                    if telegram_token and telegram_chat_id:
                        msg = f"📊 *Train Success - Model {model_name} | Dataset {dataset_name} | Prune Rate {pruning_rate}"
                        send_telegram_message(telegram_token, telegram_chat_id, msg)

                except Exception as e:
                    console.print(f"[red]✖ Error:[/red] {e} | {model_name} | {dataset_name} | {pruning_rate}")
                    
                    if telegram_token and telegram_chat_id:
                        msg = f"📊 *Train Fail - Model {model_name} | Dataset {dataset_name} | Prune Rate {pruning_rate}"
                        send_telegram_message(telegram_token, telegram_chat_id, msg)

                    continue

import os
import pandas as pd
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
import copy

import torch
import torch.nn as nn

def load_partial_state(model, pruned_state_dict):
    
    model_state_dict = model.state_dict()

    adapted_state_dict = {}

    for name, param in pruned_state_dict.items():
        if name not in model_state_dict:
            print(f"Skipping {name}: not in model")
            continue

        model_param = model_state_dict[name]

        if param.shape == model_param.shape:
            # Shapes match: keep as is
            adapted_state_dict[name] = param
        elif len(param.shape) == 4:  # likely a conv layer
            # Crop filters to match the pruned version
            out_c = param.shape[0]
            in_c = param.shape[1]
            kh, kw = param.shape[2:]
            adapted_state_dict[name] = model_param.clone()
            adapted_state_dict[name][:out_c, :in_c, :kh, :kw] = param
            print(f"Partially loaded {name} (conv): {param.shape} -> {model_param.shape}")
        elif len(param.shape) == 1:  # bias, batchnorm weight, etc.
            adapted_state_dict[name] = model_param.clone()
            adapted_state_dict[name][:param.shape[0]] = param
            print(f"Partially loaded {name} (1D): {param.shape} -> {model_param.shape}")
        else:
            print(f"Skipping {name}: shape mismatch {param.shape} vs {model_param.shape}")

    # Load the adapted state dict
    model.load_state_dict(adapted_state_dict, strict=False)
    return model



def find_filenames_with_words(words, directory):
    """
    Returns a list of filenames from the given directory that contain all specified words in the filename, regardless of order.

    Args:
        words (list): List of words to look for in filenames.
        directory (str): Path to the directory to search.

    Returns:
        list: Filenames that contain all the given words.
    """
    matched_files = []

    for filename in os.listdir(directory):
        lowercase_name = filename.lower()
        if all(word.lower() in lowercase_name for word in words):
            matched_files.append(f'{directory}/{filename}')

    return matched_files

# Searching files
list_datasets = ['cifar10']#, 'flowers102', 'food101','cifar100']
directory_path = "/media/marronedantas/HD4TB/Projects/gap-pruning/checkpoints/resnet"
actual_path = os.getcwd()
model_name = "resnet50.ra_in1k"
base_model_path = f'{directory_path}/loss_full_model_{model_name}'
batch_size = 2

dict_rate = {'5':5,'10':10,'20':20,'30':30,'40':40,'50':50,'60':60,'70':70,'80':80,'90':90,'95':95}

for dataset_name in list_datasets:

    #Loading base model
    full_path_base_model = f"{base_model_path}_{dataset_name}.pt"

    #Setting a dummy train
    dummy_train = TrainModel()
    dummy_train.set_model(id_model=model_name, num_classes=10)
    dummy_train.set_dataset(id_dataset=dataset_name, batch_size=batch_size, download=True)
    
    # Loading base model
    dummy_train.model.load_state_dict(torch.load(full_path_base_model, weights_only=False).state_dict())
    dummy_train.model.to('cuda')
    
    # Setting the dataset
    print(f"Evaluating dataset {dataset_name}")

    # List of models to generate evaluation
    words_to_search = [model_name, f'{dataset_name}_','pth']
    result = find_filenames_with_words(words_to_search, directory_path)

    # Loop over models
    for filepath in result:
        
        print(f'Processing: {filepath}')
        
        file_name = filepath.split('/')[-1]
        prune_rate = file_name.replace('.pth','').split('_')[-1]

        gap_pruning = GapPruning(model=copy.deepcopy(dummy_train.model), dataset=dummy_train.arr_dataset[0], device='cuda')
        
        std_path = f'{dataset_name}_std_devs.pth'
        
        if not os.path.isfile(std_path):
        
            gap_pruning.process_dataset()
    
        gap_pruning.compute_std_devs(file_path=std_path, flg_load=True)
        
        prunned_state = torch.load(filepath, weights_only=True)
        flg_zero_layer = gap_pruning.prune_state(prunned_state)
        print('Zero layer: ',flg_zero_layer)
        gap_pruning.model.load_state_dict(prunned_state)
        
        
        

        
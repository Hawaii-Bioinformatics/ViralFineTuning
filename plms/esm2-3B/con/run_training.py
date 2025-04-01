import pdb

from utils.model_zoo import build_avg_pooling_model, build_test_model
from utils.load_dataframes import get_df, merge_dfs
from utils.lora_converter import convert_model
from utils.multi_gpu import split_model_over_multiple_devices
import loralib as lora
from my_datasets import SiameseDataset, SupMPNDataset, sup_mpn_collate
from loss import CosineMSELoss, SupMPNLoss
from train import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from utils.save import save_model, full_save, load_model_optimizer_scheduler, parse_config_file, load_data_loader_from_pickle
from torch.cuda.amp import autocast, GradScaler
import tqdm
import datetime
import numpy as np
import os
import pickle


MODEL_TYPES = ['avg_pooling', 'test']
protocolS = ['sup_mpn', 'siamese', 'sup_mpn_hard_negative']

def resume_training_run(path):
    config_file = os.path.join(path, 'arguments.txt')
    variables = parse_config_file(config_file)
    train_loader = load_data_loader_from_pickle(path)
    print(variables)
    model_type = variables['model_type']
    base_model = variables['base_model']
    data_path = variables['data_path']
    learning_rate = variables['learning_rate']
    use_lora = variables['use_lora']
    batch_size = variables['batch_size']
    device = "cuda" #variables['device']
    # device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = variables['protocol']
    epochs = variables['epochs'] #- train_loader.dataset.epoch_count
    early_stopping = variables['early_stopping']
    load_checkpoint = variables['load_checkpoint']
    model_parallel = variables['model_parallelism']
    data_parallel = variables['data_parallelism']
    max_length = variables['max_length']
    validation_size = variables['validation_size']
    embedding_path = variables['embedding_path']
    positive_count = variables['positive_count']
    negative_count = variables['negative_count']
    mute = False
    timed_checkpoint_path = variables['timed_checkpoint_path']
    checkpoint_frequency = int(variables['checkpoint_frequency']) if variables['checkpoint_frequency'] else None

    if data_parallel:
        num_workers = torch.cuda.device_count()
    else:
        num_workers = 1

    if model_type == 'avg_pooling':
        embedding_net, tokenizer = build_avg_pooling_model(base_model)
    if model_type == 'test':
        embedding_net, tokenizer = build_test_model(base_model)
        tokenizer = lambda x: x[0] #TODO remove
    model_type = variables['model_type']
    if model_type == 'avg_pooling':
        base_model = variables['base_model']
    if torch.cuda.device_count() > 1 and model_parallel:
        if embedding_net.model_type == 'esm':
            num_gpus = torch.cuda.device_count()
            split_model_over_multiple_devices(embedding_net, [f'cuda:{i}' for i in range(num_gpus)])
        else:
            embedding_net = embedding_net.to(device)
    else:
        embedding_net = embedding_net.to(device)

    # pdb.set_trace()
    validation_fnc = validation_run
    df = get_df(data_path, mode='val')
    if protocol == 'sup_mpn':
        process_fnc = sup_mpn_process_provider(device)
        validation_dataset = SupMPNDataset(dataframe=df, max_length=max_length, embedding_path=embedding_path, pos_max_count=positive_count, neg_max_count=negative_count, early_stop=validation_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=sup_mpn_collate, num_workers=num_workers)
    if protocol == 'siamese':
        process_fnc = siamese_process_provider()
        validation_dataset = SiameseDataset(dataframe=df, max_length=max_length, embedding_path=embedding_path, early_stop=validation_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if use_lora:
        convert_model(embedding_net)
        lora_params = []
        for n, p in embedding_net.named_parameters():
            if 'lora_' in n:
                lora_params.append(p)
        print('#lora_params = ', len(lora_params))
        optimizer = optim.Adam(lora_params, lr=learning_rate)
    else:
        optimizer = optim.Adam(embedding_net.parameters(), lr=learning_rate)
    
    total_iterations = len(train_loader)
    # scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=learning_rate/10)
    scheduler = None
    embedding_net, _optimizer, scheduler, current_epoch, current_iteration = load_model_optimizer_scheduler(path, embedding_net, scheduler, optimizer, device)

    pdb.set_trace()
    save_after_epoch_path = path

    training_run(embedding_net, tokenizer, train_loader, _optimizer, process_fnc, device, epochs=epochs,
            early_stopping=early_stopping, checkpoint=load_checkpoint, use_lora=use_lora, 
            validation_fnc=validation_fnc, validation_loader=validation_loader, scheduler=scheduler,
            save_after_epoch_path=save_after_epoch_path, model_parallel=model_parallel, data_parallel=data_parallel, 
            mute=mute, timed_checkpoint_path=timed_checkpoint_path, checkpoint_frequency=checkpoint_frequency, 
            current_epoch=current_epoch, current_iteration=current_iteration)
    

def build_and_train_model(base_model,
                        model_type,
                        protocol, 
                        data_path, 
                        epochs,
                        batch_size,
                        learning_rate,
                        validation_size,
                        train_size,
                        max_length,
                        use_lora,
                        save_after_epoch_path,
                        load_checkpoint,
                        early_stopping,
                        embedding_path,
                        positive_count=None,
                        negative_count=None,
                        model_parallel=None,
                        data_parallel=None,
                        device=None,
                        cosine_annealing=False,
                        mute=False,
                        timed_checkpoint_path=None, 
                        checkpoint_frequency=None,
                        current_epoch=None):

    assert not (model_parallel and data_parallel), "cannot do model and data parallelism"
    assert model_type in MODEL_TYPES, f'unsupported model type {model_type}'
    assert protocol in protocolS, f'unsupported protocol {protocol}'
    # pdb.set_trace()
    # if save_after_epoch_path and not os.path.exists(save_after_epoch_path):
    #     os.makedirs(save_after_epoch_path)

    if model_type == 'avg_pooling':
        embedding_net, tokenizer = build_avg_pooling_model(base_model)
    if model_type == 'test':
        embedding_net, tokenizer = build_test_model(base_model)
        tokenizer = lambda x: x[0] #TODO remove

    if data_parallel:
        num_workers = torch.cuda.device_count()
    else:
        num_workers = 1

    if 'prot_t5' not in base_model and embedding_net.uses_embedding:
        assert embedding_path is not None, "model specified takes embeddings for training, please provide a path"

    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler=None
    print('!!Learning rate = ', learning_rate)
    if protocol == 'siamese':
        if use_lora:
            convert_model(embedding_net)
            lora_params = []
            for n, p in embedding_net.named_parameters():
                if 'lora_' in n:
                    lora_params.append(p)
            print('#lora_params = ', len(lora_params))
            optimizer = optim.Adam(lora_params, lr=learning_rate)
        else:
            optimizer = optim.Adam(embedding_net.parameters(), lr=learning_rate)
        # pdb.set_trace()
        df = get_df(data_path, mode='train')

        train_dataset = SiameseDataset(dataframe=df, max_length=max_length, early_stop=train_size, embedding_path=embedding_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        if cosine_annealing:
            total_iterations = len(train_loader)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=learning_rate/10)
        df = get_df(data_path, mode='val')
        validation_dataset = SiameseDataset(dataframe=df, max_length=max_length, embedding_path=embedding_path, early_stop=validation_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        process_fnc = siamese_process_provider()
        validation_fnc = validation_run
        training_run(embedding_net, tokenizer, train_loader, optimizer, process_fnc, device, epochs=epochs,
            early_stopping=early_stopping, checkpoint=load_checkpoint, use_lora=use_lora, 
            validation_fnc=validation_fnc, validation_loader=validation_loader, scheduler=scheduler,
            save_after_epoch_path=save_after_epoch_path, model_parallel=model_parallel, data_parallel=data_parallel,
            mute=mute, timed_checkpoint_path=timed_checkpoint_path, checkpoint_frequency=checkpoint_frequency, current_epoch=current_epoch)
    
    elif protocol == 'sup_mpn':
        print("in sup mpn")
        if use_lora:
            convert_model(embedding_net)
            lora_params = []
            for n, p in embedding_net.named_parameters():
                if 'lora_' in n:
                    lora_params.append(p)
            optimizer = optim.Adam(lora_params, lr=learning_rate)
        else:
            optimizer = optim.Adam(embedding_net.parameters(), lr=learning_rate)

        train_df = get_df(data_path, mode='train')
        if "prot_t5" in base_model:
            train_df["sequence"] = train_df["sequence"].apply(lambda x : " ".join(x))

        print('negative_count', negative_count)
        print('positive_count', positive_count)
        print('batch_size', batch_size)
        print('train_df\n', train_df.shape)
        print('train_df\n', train_df.head())
        
        train_dataset = SupMPNDataset(dataframe=train_df, max_length=max_length, pos_max_count=positive_count, neg_max_count=negative_count, early_stop=train_size, embedding_path=embedding_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sup_mpn_collate, num_workers=num_workers)
        if cosine_annealing:
            total_iterations = len(train_loader)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=learning_rate/10)
        val_df = get_df(data_path, mode='val')
        if "prot_t5" in base_model:
            val_df["sequence"] = train_df["sequence"].apply(lambda x : " ".join(x))
        validation_dataset = SupMPNDataset(dataframe=val_df, max_length=max_length, embedding_path=embedding_path, pos_max_count=positive_count, neg_max_count=negative_count, early_stop=validation_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=sup_mpn_collate, num_workers=num_workers)
        process_fnc = sup_mpn_process_provider(device)
        validation_fnc = validation_run
        training_run(embedding_net, tokenizer, train_loader, optimizer, process_fnc, device, epochs=epochs, model_config=model_config,
            early_stopping=early_stopping, checkpoint=load_checkpoint, use_lora=use_lora, 
            validation_fnc=validation_fnc, validation_loader=validation_loader, scheduler=scheduler,
            save_after_epoch_path=save_after_epoch_path, model_parallel=model_parallel, data_parallel=data_parallel, mute=mute,timed_checkpoint_path=timed_checkpoint_path, checkpoint_frequency=checkpoint_frequency)
    
    elif protocol == 'sup_mpn_hard_negative':
        if use_lora:
            convert_model(embedding_net)
            lora_params = []
            for n, p in embedding_net.named_parameters():
                if 'lora_' in n:
                    lora_params.append(p)
            optimizer = optim.Adam(lora_params, lr=learning_rate)
        else:
            optimizer = optim.Adam(embedding_net.parameters(), lr=learning_rate)

        df = get_df(data_path, mode='train')
        train_dataset = SupMPNDataset(dataframe=df, max_length=max_length, pos_max_count=positive_count, neg_max_count=negative_count*2, early_stop=train_size, embedding_path=embedding_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sup_mpn_collate, num_workers=num_workers)
        if cosine_annealing:
            total_iterations = len(train_loader)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=learning_rate/10)
        df = get_df(data_path, mode='val')
        validation_dataset = SupMPNDataset(dataframe=df, max_length=max_length, pos_max_count=positive_count, neg_max_count=negative_count, embedding_path=embedding_path, early_stop=validation_size)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=sup_mpn_collate, num_workers=num_workers)
        process_fnc = sup_mpn_hard_negative_process_provider(device)
        validation_fnc = validation_run
        training_run(embedding_net, tokenizer, train_loader, optimizer, process_fnc, device, epochs=epochs, 
            early_stopping=early_stopping, checkpoint=load_checkpoint, use_lora=use_lora, 
            validation_fnc=validation_fnc, validation_loader=validation_loader, scheduler=scheduler,
            save_after_epoch_path=save_after_epoch_path, model_parallel=model_parallel, data_parallel=data_parallel, mute=mute,timed_checkpoint_path=timed_checkpoint_path, checkpoint_frequency=checkpoint_frequency)
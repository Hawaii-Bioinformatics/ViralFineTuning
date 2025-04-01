import torch
import os
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import random

MEM_LIMIT = 2000000

def compute_tasks(data_point_1, second_points):
    # get a 50/50 mix of pos/neg pairs
    random.shuffle(second_points)
    results = []
    pos = 0
    neg = 0
    embedding_1, group_1, id_1 = data_point_1['embedding'][0], data_point_1['#GroupName'], data_point_1['ProteinIDs']
    for data_point_2 in second_points:
        embedding_2, group_2, id_2 = data_point_2['embedding'][0], data_point_2['#GroupName'], data_point_2['ProteinIDs']
        if id_1 == id_2:
            pass
        if group_1 == group_2:
            task = (embedding_1, group_1, embedding_2, group_2)
            results.append(task)
            pos += 1
        elif neg < pos:
            task = (embedding_1, group_1, embedding_2, group_2)
            results.append(task)
            neg += 1
        
    for data_point_2 in second_points:
        embedding_2, group_2, id_2 = data_point_2['embedding'][0], data_point_2['#GroupName'], data_point_2['ProteinIDs']
        if id_1 == id_2:
            pass
            
        if neg < pos:
            task = (embedding_1, group_1, embedding_2, group_2)
            results.append(task)
            neg += 1

    return results

def process_tasks(tasks, output_pipe):
    results = []
    for task in tasks:
        embedding_1, group_1, embedding_2, group_2 = task
        distance = cosine(embedding_1, embedding_2)
        truth = 0 if group_1 == group_2 else 1
        results.append((distance, truth))
    output_pipe.send(results)
    output_pipe.close()
    print('done')

def evaluate_spearman_correlation(data):
    prediction = []
    ground_truth = []
    num_cpus = multiprocessing.cpu_count()-1
    print(f'using {num_cpus} cpus to process {len(data)} data points')
    pool_results = []
    tasks = []
    ### compute tasks to process ####
    i = 0
    for j, data_point_1 in enumerate(data):
        if i%10000 == 0:
            print(i)
        tasks += compute_tasks(data_point_1, data[j+1:])
        i += 1
    print(f'computed {len(tasks)} tasks')
    ##################################

    ### process tasks ###
    processes = []
    output_pipes = []
    chunk_size = min(len(tasks)//num_cpus, MEM_LIMIT)
    iterations = len(tasks)//(num_cpus*chunk_size)
    print(f'using {iterations} iterations')
    last_index = 0
    for iteration in range(iterations):
        print(f'using {num_cpus} workers to complete {chunk_size} tasks each')
        for i in range(num_cpus):
            output_pipe, input_pipe = multiprocessing.Pipe()
            output_pipes.append(output_pipe)
            sublist = tasks[last_index:last_index+chunk_size]
            process = multiprocessing.Process(target=process_tasks, args=(sublist,input_pipe,))
            processes.append(process)
            process.start()
            last_index += chunk_size
        
        for output_pipe in output_pipes:
            try:
                result_chunk = output_pipe.recv()
                pool_results += result_chunk
                output_pipe.close()
                print('pipe closed')
            except:
                pass 

        print(f'{len(pool_results)} tasks processed')

        # Wait for all processes to complete
        for process in processes:
            process.join()

        print('processes closed')
        ######################

        prediction = []
        ground_truth = []
        for result in pool_results:
            distance, truth = result
            prediction.append(distance)
            ground_truth.append(truth)

    # print(prediction[:100], ground_truth[:100])
    average_correlation = spearmanr(prediction, ground_truth)      
    return average_correlation

def get_dfs(data_path):
    protein_path = os.path.join(data_path, 'tmp_data_dir/proteins.faa')
    group_path = os.path.join(data_path, 'tmp_data_dir/val_groups.tsv')
    protein_df = pd.read_csv(protein_path)
    group_df = pd.read_csv(group_path)
    return protein_df, group_df

def get_sequence_dicts(protein_df, group_df):
    # Split ProteinIDs into a list of sequences
    group_df['ProteinIDs'] = group_df['ProteinIDs'].str.split(',')
    group_df = group_df.explode('ProteinIDs')
    merged_df = group_df.merge(protein_df, left_on='ProteinIDs', right_on='id', how='left')
    merged_df = merged_df.drop(columns='id')
    df_cleaned = merged_df.dropna(subset=['sequence'])
    return df_cleaned.to_dict(orient='records')

def evaluate_test_data(model, tokenizer, data_path, downsample=None, random_sample=None):
    path = os.path.join(data_path, 'tmp_data_dir/val.tsv')
    protein_sequence_dicts = pd.read_csv(path)
    if random_sample:
        num_rows_in_df = protein_sequence_dicts.shape[0]
        num_rows_to_drop = int(num_rows_in_df*(1-random_sample))
        indices_to_drop = random.sample(range(num_rows_in_df), num_rows_to_drop)
        protein_sequence_dicts = protein_sequence_dicts.drop(indices_to_drop)
    protein_sequence_dicts = protein_sequence_dicts.to_dict(orient='records')
    ### down sample ###
    if downsample:
        group_counts = defaultdict(int)
        i = 0
        for j in range(len(protein_sequence_dicts)):
            item = protein_sequence_dicts[i]
            group = item['#GroupName']
            if group_counts[group] >= downsample:
                protein_sequence_dicts = protein_sequence_dicts[:i] + protein_sequence_dicts[i+1:] #remove i
            else:
                group_counts[group] += 1
                i += 1
        print(f'data is now {len(protein_sequence_dicts)}')
    ###################
    model.eval()
    # if torch.cuda.is_available():
    model = model.to('cuda:0')
    model.base_model._init_layer_devices()
    i = 0
    for protein in protein_sequence_dicts:
        if i % 5000 == 0:
            print(i)
        tokens = tokenizer([protein['sequence']])
        with torch.no_grad():
            tokens = tokens.to('cuda:0')
            embedding_val = model(tokens)
            embedding_val = embedding_val.cpu().detach().numpy()
        protein['embedding'] = embedding_val
        i += 1
    output_file = 'data.pkl'

    # Save the data to a pickle file
    with open(output_file, 'wb') as file:
        pickle.dump(protein_sequence_dicts, file)
    spearman = evaluate_spearman_correlation(protein_sequence_dicts)
    print(f'Spearman correlation is: {spearman}')
    return spearman

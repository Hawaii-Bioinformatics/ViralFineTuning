import pandas as pd
import os

def get_df(data_path, mode='test'):
    group_path = os.path.join(data_path, f'tmp_data_dir/{mode}.tsv')
    df = pd.read_csv(group_path)
    return df

def merge_dfs(protein_df, group_df):
    # Split ProteinIDs into a list of sequences
    group_df['ProteinIDs'] = group_df['ProteinIDs'].str.split(',')
    group_df['ProteinIDs'] = group_df['ProteinIDs'].apply(lambda x: list(set(x))) #remove duplicates
    group_df = group_df[group_df['ProteinIDs'].apply(lambda x: len(x) > 1)] #ensure no lone pairs
    group_df = group_df.explode('ProteinIDs')
    merged_df = group_df.merge(protein_df, left_on='ProteinIDs', right_on='id', how='left')
    merged_df = merged_df.drop(columns='id')
    df_cleaned = merged_df.dropna(subset=['sequence'])
    group_counts = df_cleaned['#GroupName'].value_counts()
    groups_to_remove = group_counts[group_counts == 1].index
    df_cleaned = df_cleaned[~df_cleaned['#GroupName'].isin(groups_to_remove)]
    return df_cleaned


def parse_faa_file(file_path):
    sequences = []
    current_sequence = {"id": "", "description": "", "sequence": ""}
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence["id"]:
                    sequences.append(current_sequence.copy())
                parts = line.split(" ", 2)
                current_sequence["id"] = parts[1][:]
                current_sequence["description"] = parts[2]
                current_sequence["sequence"] = ""
            else:
                current_sequence["sequence"] += line
    
    if current_sequence["id"]:
        sequences.append(current_sequence.copy())
    
    df = pd.DataFrame(sequences)
    return df

def parse_tsv_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df
from cgi import test
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from utils.load_dataframes import parse_faa_file, parse_tsv_file, merge_dfs

def check_dfs(train_df, test_df):
    train_proteins = set()
    for index, row in train_df.iterrows():
        protein_id = row['ProteinIDs']
        train_proteins.add(protein_id)

    for index, row in test_df.iterrows():
        protein_id = row['ProteinIDs']
        assert protein_id not in train_proteins

    train_groups = set()
    for index, row in train_df.iterrows():
        group_id = row['#GroupName']
        train_groups.add(group_id)

    for index, row in test_df.iterrows():
        group_id = row['#GroupName']
        assert group_id not in train_groups 

def create_dataset_folders(data_path, dest_path, split_ratio):
    directory_path = os.path.join(dest_path, "tmp_data_dir")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    train_path = os.path.join(directory_path, "train.tsv")
    val_path = os.path.join(directory_path, "val.tsv")
    test_path = os.path.join(directory_path, "test.tsv")

    train_groups_df, val_groups_df, test_groups_df = create_train_test_val_dfs(data_path, split_ratio)
    train_groups_df.to_csv(train_path, index=False)
    val_groups_df.to_csv(val_path, index=False)
    test_groups_df.to_csv(test_path, index=False)

def create_train_test_val_dfs(data_path, split_ratio):
    faa_data_location = os.path.join(data_path, 'vog.all.proteins.faa')
    tsv_data_location = os.path.join(data_path, "vog.members.tsv")
    sequence_df = parse_faa_file(faa_data_location)
    groups_df = parse_tsv_file(tsv_data_location)
    merged_df = merge_dfs(sequence_df, groups_df)

    train_df, test_df = create_train_test_split(merged_df, split_ratio)
    check_dfs(train_df, test_df)
    train_df, val_df = create_train_test_split(train_df, split_ratio)
    check_dfs(train_df, val_df)

    return train_df, val_df, test_df

def create_train_test_split(data_frame, test_size, random_state=None):
    print(len(data_frame))
    groups = data_frame['ProteinIDs'].value_counts()
    groups_to_remove = groups[groups > 1].index #remove proteins belonging to multipel homologs
    data_frame = data_frame[~data_frame['ProteinIDs'].isin(groups_to_remove)]
    
    # Get unique values of column A
    unique_values = data_frame['#GroupName'].unique()
    df = data_frame
    # Split unique values into train and test
    unique_train, unique_test = train_test_split(unique_values, test_size=test_size, random_state=42)

    # Create train and test DataFrames
    train_df = df[df['#GroupName'].isin(unique_train)]
    test_df = df[df['#GroupName'].isin(unique_test)]

    additional_rows = data_frame[data_frame['ProteinIDs'].isin(groups_to_remove)]
    test_df = pd.concat([test_df, additional_rows])

    return train_df, test_df

def compute_embedding_function(model):
    def helper(input):
        with torch.no_grad():
            return model(input)

    return helper


def add_embeddings(df, model):
    if torch.cuda.is_available():
        model.cuda()
    func = compute_embedding_function(model)
    new_row = df.apply(func, axis=1)
    df = df.append(new_row, ignore_index=True)
    return df
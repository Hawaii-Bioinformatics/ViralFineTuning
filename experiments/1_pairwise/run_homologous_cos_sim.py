import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import itertools
import argparse

from tqdm import tqdm

import pdb

def get_seq_pair_sim(seq1_id, seq2_id, emb_dir):
    """
    Compute cosine similarity between two sequences using averaged ESM2 embeddings.
    Supports .pt or .npy files.
    """
    try:
        # pdb.set_trace()
        seq1_pooled_emb = torch.mean(torch.load(f'{emb_dir}/{seq1_id}.pt'), dim=1)
        seq2_pooled_emb = torch.mean(torch.load(f'{emb_dir}/{seq2_id}.pt'), dim=1)
        
        sim_score = F.cosine_similarity(seq1_pooled_emb, seq2_pooled_emb).item()
        return sim_score

    except Exception as e:
        # Could log error here
        return None


def get_vog_seq_sims(vog, data, emb_dir):
    """
    Compute pairwise cosine similarities within a vog.
    """
    seq_ids = data.loc[data['group'] == vog, 'protein_id'].tolist()
    all_pairs = list(itertools.combinations(seq_ids, 2))

    pair_df = pd.DataFrame(all_pairs, columns=['seq1_id', 'seq2_id'])
    pair_df['group'] = vog

    pair_df['cos_sim'] = pair_df.apply(
        lambda row: get_seq_pair_sim(row['seq1_id'], row['seq2_id'], emb_dir),
        axis=1
    )
    return pair_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, required=True, help="Path to directory with embeddings for the model")

    args = parser.parse_args()

    emb_dir = args.emb_dir

    data = pd.read_pickle("vog_seq_test_df.pkl")  # Modify as needed

    pdb.set_trace()
    vogs = list(data.group.unique())

    results = []
    for vog in tqdm(vogs, desc="Computing pairwise cosine similarity per VOG", dynamic_ncols=True):
        df = get_vog_seq_sims(vog, data, emb_dir)  # df is returned per VOG
        results.append(df)

    results_df = pd.concat(results, ignore_index=True)
    
    pdb.set_trace()

    # Save or further process `results`
    csv_path = f"exp_results/cos_sim/homologous_pairwise_cosine_sims.csv"
    results_df.to_csv(csv_path, index=False)

    print("Done. Results saved to:", csv_path)
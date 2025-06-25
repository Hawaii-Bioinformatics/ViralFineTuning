import os
import sys
import random
import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score
from multiprocessing import Pool
from functools import partial


# -------------- Clustering Functions --------------

def append_to_tsv(output_filename, title, iteration, silhouette_avg):
    tsv_file = f"exp_results/hdbscan/{title}/{output_filename}"
    os.makedirs(os.path.dirname(tsv_file), exist_ok=True)
    if not os.path.exists(tsv_file):
        with open(tsv_file, "w") as f:
            f.write("Iteration\tTitle\tSilhouette_Avg\n")
    with open(tsv_file, "a") as f:
        f.write(f"{iteration}\t{title}\t{silhouette_avg:.4f}\n")


def cluster_with_hdbscan(X, title, iteration, output_filename, min_cluster_size=10, min_samples=5):
    # print(f'[Iteration {iteration}] Clustering for: {title}')
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        algorithm='best',
        alpha=1.0,
        cluster_selection_epsilon=1.0,
        cluster_selection_method='eom',
    )
    cluster_labels = hdbscan_clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f'[Iteration {iteration} , {title}] silhouette_avg = {silhouette_avg:.4f}')
    append_to_tsv(output_filename, title, iteration, silhouette_avg)
    return cluster_labels, silhouette_avg


# -------------- Per-Iteration Worker --------------

def run_iteration(i, pkl_path, title):
    try:
        data = pd.read_pickle(pkl_path)
        # Ensure group names are cached
        unique_vogs = data['group'].unique().tolist()

        random.seed(i)  # ensure reproducibility
        random_vogs = random.sample(unique_vogs, 500)
        rand_500_df = data[data['group'].isin(random_vogs)].copy()

        if rand_500_df.empty or 'emb' not in rand_500_df.columns:
            print(f'[Iteration {i}] No embeddings found')
            return

        emb_stack = np.vstack(rand_500_df['emb'])
        output_filename = f"out_{i}.tsv"

        cluster_with_hdbscan(
            emb_stack,
            title=title,
            iteration=i,
            output_filename=output_filename
        )
    except Exception as e:
        print(f'[Iteration {i}] ERROR: {e}')


if __name__ == "__main__":
    i = int(sys.argv[1])
    pkl_path = sys.argv[2]
    if len(sys.argv) > 3:
        title = sys.argv[3]
    else:
        # Derive title from filename: /path/to/viral-esm-con_mean_emb.pkl -> viral-esm-con
        title = os.path.basename(pkl_path).split('_mean_emb.pkl')[0]
    run_iteration(i, pkl_path, title)


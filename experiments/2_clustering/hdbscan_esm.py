import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
import hdbscan
from sklearn.metrics import silhouette_score
import os

# directories
data_dir = "/data/rajan/vog"
msa_fasta_dir = f"{data_dir}/fasta"
pkl_dir = f"{data_dir}/data-pkls"
plot_dir = "plots/new/clustering"


# pdb.set_trace()
# Load the dataframe with embeddings
print('loading dataframe with embeddings')
data = pd.read_pickle(open(f"./clustering_df_w_emb_esm.pkl", "rb"))
print('total unique labels/grp_ids:', len(data.align_id.unique()))
print("selecting random 500 labels/group_ids and their respective rows")
# Select random 500 align_ids
random_align_ids = random.sample(data['align_id'].unique().tolist(), 500)

# Select rows from data belonging to those random 500 align_ids
rand_500_df = data[data['align_id'].isin(random_align_ids)].copy()

# Create X values (embeddings)
esm2_3B_X_vals = np.vstack(rand_500_df['esm_pt'])
esm2_3B_cls_X_vals = np.vstack(rand_500_df['esm_cls_lora'])
esm2_3B_mlm_X_vals = np.vstack(rand_500_df['esm_mlm_lora'])
esm2_3B_cont_X_vals = np.vstack(rand_500_df['esm_cont'])

# Function to append data to a TSV file
def append_to_tsv(iteration, title, silhouette_avg):
    tsv_file = f"results/esm/{output_filename}"
    if not os.path.exists(tsv_file):
        with open(tsv_file, "w") as f:
            f.write("Iteration\tTitle\tSilhouette_Avg\n")
    with open(tsv_file, "a") as f:
        f.write(f"{iteration}\t{title}\t{silhouette_avg:.2f}\n")

def cluster_with_hdbscan(X, title='ESM2-3B', iteration=0, min_cluster_size=10, min_samples=5):
    print('clustering for:', title)
    
    # Apply HDBSCAN clustering directly to the embeddings
    # settings based on optimize run - check optimize.log
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
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print('silhouette_avg = ', silhouette_avg)
    
    # Append silhouette score to TSV file
    append_to_tsv(iteration, title, silhouette_avg)
    
    return cluster_labels, silhouette_avg

iteration = str(sys.argv[1])
output_filename = str(sys.argv[2])

cluster_labels, silhouette_avg = cluster_with_hdbscan(esm2_3B_X_vals, "ESM2-3B-Pre-trained", iteration=iteration)
cluster_labels, silhouette_avg = cluster_with_hdbscan(esm2_3B_cont_X_vals, title='ESM2-3B-Contrastive', iteration=iteration)
cluster_labels, silhouette_avg = cluster_with_hdbscan(esm2_3B_cls_X_vals, title='ESM2-3B-Classification', iteration=iteration)
cluster_labels, silhouette_avg = cluster_with_hdbscan(esm2_3B_mlm_X_vals, title='ESM2-3B-MLM', iteration=iteration)

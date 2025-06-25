import itertools 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from scipy import stats
import numpy as np
import seaborn as sns
from scipy.spatial import distance
import argparse
import os


# configs
layer_esm2_3B = 36


def compute_sim_score(seq1_id, seq2_id, emb_idx1, emb_idx2, main_dir):
    """
    Computes and returns the cosine sim between the two residues using tokenwise embeddings
    aa_idx -> emb_idx
    """
    pt_emb_dir = os.path.join(main_dir, 'pre-trained')
    ft_cls_emb_dir = os.path.join(main_dir, 'ft_cls')
    ft_mlm_emb_dir = os.path.join(main_dir, 'ft_mlm')
    ft_cont_emb_dir = os.path.join(main_dir, 'ft_con')
    try:
        # pretrained
        aa1_pt_emb = torch.load(f'{pt_emb_dir}/{seq1_id}.pt')["representations"][layer_esm2_3B][emb_idx1] 
        aa2_pt_emb = torch.load(f'{pt_emb_dir}/{seq2_id}.pt')["representations"][layer_esm2_3B][emb_idx2]
        # lora cls
        aa1_ft_cls_emb = torch.load(f'{ft_cls_emb_dir}/{seq1_id}.pt', map_location=torch.device('cpu'))[1:-1][emb_idx1]
        aa2_ft_cls_emb = torch.load(f'{ft_cls_emb_dir}/{seq2_id}.pt', map_location=torch.device('cpu'))[1:-1][emb_idx2]
        # # lora mlm
        aa1_ft_mlm_emb = torch.load(f"{ft_mlm_emb_dir}/{seq1_id}.pt", map_location=torch.device('cpu'))[1:-1][emb_idx1]  # using [1:-1] to remove padding
        aa2_ft_mlm_emb = torch.load(f"{ft_mlm_emb_dir}/{seq2_id}.pt", map_location=torch.device('cpu'))[1:-1][emb_idx2] 
        # contrastive
        aa1_ft_cont_emb = torch.from_numpy(np.load(f"{ft_cont_emb_dir}/{seq1_id}.npy")[1:-1])[emb_idx1]  # using [1:-1] to remove padding
        aa2_ft_cont_emb = torch.from_numpy(np.load(f"{ft_cont_emb_dir}/{seq2_id}.npy")[1:-1])[emb_idx2] 
    except Exception as e:
        # print(e, file=open('sim_errors.txt', 'a'))
        return None, None, None, None, None, None, None, None
    result = (F.cosine_similarity(aa1_pt_emb.unsqueeze(0), aa2_pt_emb.unsqueeze(0)).item(),
            F.cosine_similarity(aa1_ft_cls_emb.unsqueeze(0), aa2_ft_cls_emb.unsqueeze(0)).item(),
            F.cosine_similarity(aa1_ft_mlm_emb.unsqueeze(0), aa2_ft_mlm_emb.unsqueeze(0)).item(),
            F.cosine_similarity(aa1_ft_cont_emb.unsqueeze(0), aa2_ft_cont_emb.unsqueeze(0)).item(),
            stats.spearmanr(aa1_pt_emb, aa2_pt_emb).statistic,
            stats.spearmanr(aa1_ft_cls_emb, aa2_ft_cls_emb).statistic,
            stats.spearmanr(aa1_ft_mlm_emb, aa2_ft_mlm_emb).statistic,
            stats.spearmanr(aa1_ft_cont_emb, aa2_ft_cont_emb).statistic)
    return result


def get_random_valid_aa_idx(aa_idx2emb_idx, cons_col_list, other_random_aa_idx=None):
    np.random.seed(8)
    if other_random_aa_idx is None:
        return np.random.choice([aa_idx for aa_idx, emb_idx in enumerate(aa_idx2emb_idx) if emb_idx is not None and aa_idx not in cons_col_list])
    else:
        return np.random.choice([aa_idx for aa_idx, emb_idx in enumerate(aa_idx2emb_idx) if emb_idx is not None and aa_idx not in cons_col_list and abs(aa_idx - other_random_aa_idx)>20])

error_emb_msa = []
def get_non_cons_res_sim(main_dir, data, filtered_msa2cons_col):
    non_cons_res_sim_list = []
    total_seqs=0
    for align_id, val in filtered_msa2cons_col.items():
        if total_seqs >= 20000:
            print(total_seqs)
            break
        align_df = data[data['align_id'] == align_id].copy()
        if align_df.empty:
            print('skipping:', align_id)
            continue
        print('processing ', align_id)
        cons_col_list = val['cons_col']
        msa_seq_len = len(align_df.seq_str.iloc[0])
        num_seqs = len(align_df)
        total_seqs+=num_seqs
        print('num seqs = ', num_seqs)
        print('total_seqs processed = ', total_seqs)
        seq_pairs = list(itertools.combinations(align_df.seq_id, 2))
        for seq1_id, seq2_id in seq_pairs[:4]:
            seq1_row = align_df.loc[align_df['seq_id'] == seq1_id]
            seq2_row = align_df.loc[align_df['seq_id'] == seq2_id]
            seq1_str = seq1_row.seq_str.item()
            seq2_str = seq2_row.seq_str.item()
            # pick random AA from each seq_id
            random_aa1_idx = get_random_valid_aa_idx(seq1_row.aa_idx2emb_idx.item(), cons_col_list)
            random_aa2_idx = get_random_valid_aa_idx(seq2_row.aa_idx2emb_idx.item(), cons_col_list, random_aa1_idx)
            aa1 = seq1_str[random_aa1_idx]
            aa2 = seq2_str[random_aa2_idx]
            emb_idx1 = seq1_row.aa_idx2emb_idx.item()[random_aa1_idx]
            emb_idx2 = seq2_row.aa_idx2emb_idx.item()[random_aa2_idx]
            pt_score, ft_cls_score, ft_mlm_score, ft_cont_score, pt_sp_rank, ft_cls_sp_rank, ft_mlm_sp_rank, ft_cont_sp_rank  = compute_sim_score(seq1_id, seq2_id, emb_idx1, emb_idx2, main_dir)
            hdist = round(distance.hamming(list(seq1_str), list(seq2_str)) * msa_seq_len)
            if pt_score is None or ft_cls_score is None or ft_mlm_score is None or ft_cont_score is None:
                if align_id not in error_emb_msa:
                    error_emb_msa.append(align_id)
            else:
                non_cons_res_sim_list.append((
                    align_id, seq1_id, seq2_id, aa1, aa2, pt_score, ft_cls_score,ft_mlm_score, ft_cont_score, pt_sp_rank, ft_cls_sp_rank, ft_mlm_sp_rank, ft_cont_sp_rank, hdist))
    non_cons_res_sim_df = pd.DataFrame(
        non_cons_res_sim_list,
        columns=['align_id', 'seq1_id', 'seq2_id', 'aa1', 'aa2', 'pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score', 'pt_sp_rank', 'ft_cls_rank', 'ft_mlm_sp_rank', 'ft_cont_sp_rank', 'hdist'])
    return non_cons_res_sim_df


def get_align_mean_sim_scores(esm_non_cons_res_sim_df):
    """
    Get mean of all pairwise sim scores per alignment using MSA embd
    """
    esm_non_cons_col_mean_sim_df = esm_non_cons_res_sim_df.groupby('align_id').agg({
        'pt_score': 'mean',
        'ft_cls_score': 'mean',
        'ft_mlm_score': 'mean',
        'ft_cont_score': 'mean',
        'pt_sp_rank': 'mean',
        'ft_cls_rank': 'mean',
        'ft_mlm_sp_rank': 'mean',
        'ft_cont_sp_rank': 'mean'
    }).reset_index()
    return esm_non_cons_col_mean_sim_df


def save_kde(df, title, type):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df["pt_score"],
                color='red', label='pre-trained', fill=True, ax=ax, bw_adjust=2)
    sns.kdeplot(data=df["ft_cls_score"],
                color='blue', label='fine-tuned (Classification-LoRA)', fill=True, ax=ax, bw_adjust=2)
    sns.kdeplot(data=df["ft_mlm_score"],
                color='gold', label='fine-tuned (MLM-LoRA)', fill=True, ax=ax, bw_adjust=2)
    sns.kdeplot(data=df["ft_cont_score"],
                color='green', label='fine-tuned (Contrastive)', fill=True, ax=ax, bw_adjust=2)
    ax.legend()
    sns.move_legend(ax, "upper left")
    plt.title(title)
    plt.xlabel('Cosine Similarity of Non-Conserved Sites')
    plt.savefig(f'{plot_dir}/esm-kde-non-cons-col-sim-{type}.png')
    plt.tight_layout()
    plt.show()

def save_kde_sprank(df, title, type):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df["pt_sp_rank"],
                color='red', label='pre-trained', fill=True, ax=ax, bw_adjust=2)
    sns.kdeplot(data=df["ft_cls_rank"],
                color='blue', label='fine-tuned (Classification-LoRA)', fill=True, ax=ax, bw_adjust=2)
    sns.kdeplot(data=df["ft_mlm_sp_rank"],
                color='gold', label='fine-tuned (MLM-LoRA)', fill=True, ax=ax, bw_adjust=2)
    sns.kdeplot(data=df["ft_cont_sp_rank"],
                color='green', label='fine-tuned (Contrastive)', fill=True, ax=ax, bw_adjust=2)
    ax.legend()
    sns.move_legend(ax, "upper left")
    plt.title(title)
    plt.xlabel('Spreaman R of Non-Conserved Sites')
    plt.savefig(f'{plot_dir}/esm-kde-non-cons-col-sprank-{type}.png')
    plt.tight_layout()
    plt.show()

def save_violin_plot(df, title='ESM2', label='Similarity Score'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score']], inner='quartile')
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-non-cons-col-sim-violin-plot.png')
    plt.show()

def save_violin_plot_sprank(df, plot_dir, title='ESM2', label='Spearman R'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['pt_sp_rank', 'ft_cls_rank', 'ft_mlm_sp_rank', 'ft_cont_sp_rank']], inner='quartile')
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-non-cons-col-sprank-violin-plot.png')
    plt.show()

def save_bean_plot(df, plot_dir, title='ESM2', label='Similarity Score'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score']], inner='stick', density_norm='count')
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-non-cons-col-sim-bean-plot.png')
    plt.show()

def save_swarm_plot(df, plot_dir, title='ESM2', label='Similarity Score'):
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=df[['pt_score', 'ft_cls_score', 'ft_cont_score']], size=5)
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-non-cons-col-sim-swarm-plot.png')
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, required=True, help="Path to directory with MSA embeddings")
    parser.add_argument("--plot_dir", type=str, required=True, help="Path to directory where plots will be saved")

    args = parser.parse_args()

    emb_dir = args.emb_dir
    plot_dir = args.plot_dir

    data = pickle.load(open("new_sel_msa_w_cons_col_df.pkl", "rb"))
    print('unique align_ids: ', len(data.align_id.unique()))

    msa2cons_col = pickle.load(open("msa2cons_col.pkl", "rb"))
    unique_align_ids = data['align_id'].unique().tolist()

    # Filter the msa2cons_col dictionary to only include these unique align_ids
    filtered_msa2cons_col = {key: msa2cons_col[key] for key in unique_align_ids if key in msa2cons_col}
    # Print or check the filtered dictionary
    print('unique keys: ', len(filtered_msa2cons_col.keys()))

    esm_non_cons_res_sim_df = get_non_cons_res_sim(emb_dir, data, filtered_msa2cons_col)

    # esm_non_cons_res_sim_df.to_pickle(open('esm_non_cons_sim_df.pkl', "wb"))
    esm_non_cons_res_sim_df = pd.read_pickle(open("esm_non_cons_sim_df.pkl", "rb"))
    print(esm_non_cons_res_sim_df)
    print(esm_non_cons_res_sim_df.columns)


    print('pretrained:', esm_non_cons_res_sim_df.pt_score.min(), esm_non_cons_res_sim_df.pt_score.max())
    print('ft_cls:', esm_non_cons_res_sim_df.ft_cls_score.min(), esm_non_cons_res_sim_df.ft_cls_score.max())
    print('ft_mlm:', esm_non_cons_res_sim_df.ft_mlm_score.min(), esm_non_cons_res_sim_df.ft_mlm_score.max())
    print('ft_cont:', esm_non_cons_res_sim_df.ft_cont_score.min(), esm_non_cons_res_sim_df.ft_cont_score.max())

    esm_non_cons_col_mean_sim_df = get_align_mean_sim_scores(esm_non_cons_res_sim_df)

    selected_columns = ['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score', 'pt_sp_rank', 'ft_cls_rank', 'ft_mlm_sp_rank', 'ft_cont_sp_rank']

    # Compute statistics: min, max, mean
    stats = esm_non_cons_col_mean_sim_df[selected_columns].agg(['min', 'max', 'mean']).transpose()

    # Print the statistics
    print(f"Statistics for selected columns:\n{stats}")

    # save plots
    save_kde(esm_non_cons_res_sim_df, plot_dir, title='ESM2-3B', type='not-mean')
    save_kde(esm_non_cons_col_mean_sim_df, plot_dir, title='ESM2-3B', type='mean')

    save_kde_sprank(esm_non_cons_res_sim_df, plot_dir, title='ESM2-3B', type='not-mean')
    save_kde_sprank(esm_non_cons_col_mean_sim_df, plot_dir, title='ESM2-3B', type='mean')

    save_violin_plot(esm_non_cons_col_mean_sim_df, plot_dir, 'ESM2-3B')
    save_violin_plot_sprank(esm_non_cons_col_mean_sim_df, plot_dir, 'ESM2-3B')



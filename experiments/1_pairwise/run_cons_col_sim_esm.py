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
import os
import argparse


# configs
layer_esm2_3B = 36


def compute_sim_score(seq1_id, seq2_id, emb_idx1, emb_idx2, main_dir):
    """
    Computes and returns the cosine sim between the two residues using tokenwise embeddings
    aa_idx -> emb_idx
    """
    # pdb.set_trace()
    pt_emb_dir = os.path.join(main_dir, 'pre-trained')
    ft_cls_emb_dir = os.path.join(main_dir, 'ft_cls')
    ft_mlm_emb_dir = os.path.join(main_dir, 'ft_mlm')
    ft_cont_emb_dir = os.path.join(main_dir, 'contrastive')
    # ft_mlm_emb_dir = os.path.join(main_dir, 'ft_mlm')
    try:
        # pretrained
        aa1_pt_emb = torch.load(f'{pt_emb_dir}/{seq1_id}.pt')["representations"][layer_esm2_3B][emb_idx1] 
        aa2_pt_emb = torch.load(f'{pt_emb_dir}/{seq2_id}.pt')["representations"][layer_esm2_3B][emb_idx2]
        # lora cls
        aa1_ft_cls_emb = torch.load(f'{ft_cls_emb_dir}/{seq1_id}.pt', map_location=torch.device('cpu'))[1:-1][emb_idx1]
        aa2_ft_cls_emb = torch.load(f'{ft_cls_emb_dir}/{seq2_id}.pt', map_location=torch.device('cpu'))[1:-1][emb_idx2]
        # lora mlm
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
    # pdb.set_trace()
    # print(result)
    return result


error_emb_msa = []
def get_cons_col_sim(main_dir, data, filtered_msa2cons_col ):
    cons_col_sim_list = []
    total_seqs=0
    for align_id, val in list(filtered_msa2cons_col.items()):
        cons_col_list = val['cons_col']
        if total_seqs >= 20000:
            print(total_seqs)
            break
        print(align_id, cons_col_list)
        align_df = data[data['align_id'] == align_id]
        if align_df.empty:
            print('skipping:', align_id)
            continue
        print('processing ', align_id)
        seq_ids = align_df['seq_id'].tolist()
        msa_seq_len = len(align_df.seq_str.iloc[0])
        for col in cons_col_list:
            print('processing col:', col)
            valid_seq_ids = [seq_id for seq_id in seq_ids if align_df.loc[align_df['seq_id'] == seq_id, 'aa_idx2emb_idx'].item()[col] is not None]
            total_seqs+=len(valid_seq_ids)
            print('len valid_seq_ids:', len(valid_seq_ids))
            if not valid_seq_ids:
                continue
            seq_pairs = list(itertools.combinations(valid_seq_ids, 2))
            print('len seq_pairs:', len(seq_pairs))
            for seq1_id, seq2_id in seq_pairs:
                seq1_str = align_df.loc[align_df['seq_id'] == seq1_id, 'seq_str'].item()
                seq2_str = align_df.loc[align_df['seq_id'] == seq2_id, 'seq_str'].item()
                aa1 = seq1_str[col]
                aa2 = seq2_str[col]
                emb_idx1 = align_df.loc[align_df['seq_id'] == seq1_id, 'aa_idx2emb_idx'].item()[col]
                emb_idx2 = align_df.loc[align_df['seq_id'] == seq2_id, 'aa_idx2emb_idx'].item()[col]
                pt_score, ft_cls_score, ft_mlm_score, ft_cont_score, pt_sp_rank, ft_cls_rank, ft_mlm_sp_rank, ft_cont_sp_rank  = compute_sim_score(seq1_id, seq2_id, emb_idx1, emb_idx2, main_dir)
                hdist = round(distance.hamming(list(seq1_str), list(seq2_str)) * msa_seq_len)
                if pt_score is None or ft_cls_score is None or ft_mlm_score is None:
                    if align_id not in error_emb_msa:
                        error_emb_msa.append(align_id)
                else:
                    cons_col_sim_list.append((
                        align_id, seq1_id, seq2_id, col, aa1, aa2, pt_score, ft_cls_score, ft_mlm_score, ft_cont_score, pt_sp_rank, ft_cls_rank, ft_mlm_sp_rank, ft_cont_sp_rank, hdist))
                    # print(cons_col_sim_list)
        print('total_seqs processed = ', total_seqs)
    cons_col_sim_df = pd.DataFrame(
        cons_col_sim_list,
        columns=['align_id', 'seq1_id', 'seq2_id', 'pos', 'seq1_AA', 'seq2_AA', 'pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score', 'pt_sp_rank', 'ft_cls_rank', 'ft_mlm_sp_rank', 'ft_cont_sp_rank', 'hdist'])
    # with open('cons_col_missing_msa_emb.txt', 'w') as f:
    #     for msa in error_emb_msa:
    #         f.write(f"{msa}\n")
    return cons_col_sim_df


# esm_cons_col_sim_df = get_cons_col_sim(esm_msa_emb_dir)
# esm_cons_col_sim_df.to_pickle('esm_cons_col_sim_df.pkl')
# esm_cons_col_sim_df = pd.read_pickle(open('esm_cons_col_sim_df.pkl', 'rb'))

def get_align_mean_sim_scores(esm2_3B_per_align_scores):
    """
    Get mean of all pairwise sim scores per alignment using MSA embd
    """
    esm2_3B_per_align_scores = esm2_3B_per_align_scores.groupby('align_id').agg({
        'pt_score': 'mean',
        'ft_cls_score': 'mean',
        'ft_mlm_score': 'mean',
        'ft_cont_score': 'mean',
        'pt_sp_rank': 'mean',
        'ft_cls_rank': 'mean',
        'ft_mlm_sp_rank': 'mean',
        'ft_cont_sp_rank': 'mean'
    }).reset_index()
    return esm2_3B_per_align_scores


def save_kde(df, plot_dir, title, type='mean'):
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
    plt.xlabel('Cosine Similarity of Conserved Sites')
    plt.savefig(f'{plot_dir}/esm-kde-cons-col-sim-{type}.png')
    plt.tight_layout()
    plt.show()


def save_kde_sprank(df, plot_dir, title, type='mean'):
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
    plt.xlabel('Spearman R of Conserved Sites')
    plt.savefig(f'{plot_dir}/esm-kde-cons-col-sprank-{type}.png')
    plt.tight_layout()
    plt.show()


def save_violin_plot(df, plot_dir, title='ESM2', label='Similarity Score'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score']], inner='quartile')
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-cons-col-sim-violin-plot.png')
    plt.show()


def save_violin_sprank_plot(df, plot_dir, title='ESM2', label='Spearman R'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['pt_sp_rank', 'ft_cls_rank', 'ft_mlm_sp_rank', 'ft_cont_sp_rank']], inner='quartile')
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-cons-col-sprank-violin-plot.png')
    plt.show()


def save_bean_plot(df, plot_dir, title='ESM2', label='Similarity Score'):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score']], inner='stick', density_norm='count')
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-cons-col-sim-bean-plot.png')
    plt.show()


def save_swarm_plot(df, plot_dir, title='ESM2', label='Similarity Score'):
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=df[['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score']], size=5)
    plt.title(title)
    plt.ylabel(label)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['pre-trained', 'fine-tuned (Classification-LoRA)', 'fine-tuned (MLM-LoRA)', 'fine-tuned (Contrastive)'])
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/esm-cons-col-sim-swarm-plot.png')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, required=True, help="Path to directory with MSA embeddings")
    parser.add_argument("--plot_dir", type=str, required=True, help="Path to directory where plots will be saved")

    args = parser.parse_args()

    # plot_dir = "plots/conserved"
    # emb_dir = "/home/rsawhney/emb"
    # esm_msa_emb_dir = f"{emb_dir}/esm/msa/"

    emb_dir = args.emb_dir
    plot_dir = args.plot_dir

    data = pickle.load(open("new_sel_msa_w_cons_col_df.pkl", "rb"))

    msa2cons_col = pickle.load(open("msa2cons_col.pkl", "rb"))
    unique_align_ids = data['align_id'].unique()
    # Filter the msa2cons_col dictionary to only include these unique align_ids
    filtered_msa2cons_col = {key: msa2cons_col[key] for key in unique_align_ids if key in msa2cons_col}
    # Print or check the filtered dictionary
    print('unique keys: ', len(filtered_msa2cons_col.keys()))
    # pdb.set_trace()


    pdb.set_trace()
    esm_cons_col_sim_df = get_cons_col_sim(emb_dir, data, filtered_msa2cons_col)
    esm_cons_col_sim_df.to_pickle('esm_cons_col_sim_df.pkl')

    print(esm_cons_col_sim_df)
    print(esm_cons_col_sim_df.columns)
    print('pretrained:', esm_cons_col_sim_df.pt_score.min(), esm_cons_col_sim_df.pt_score.max())
    print('ft_cls:', esm_cons_col_sim_df.ft_cls_score.min(), esm_cons_col_sim_df.ft_cls_score.max())
    print('ft_mlm:', esm_cons_col_sim_df.ft_mlm_score.min(), esm_cons_col_sim_df.ft_mlm_score.max())
    print('ft_cont:', esm_cons_col_sim_df.ft_cont_score.min(), esm_cons_col_sim_df.ft_cont_score.max())

    esm_cons_col_mean_sim_df = get_align_mean_sim_scores(esm_cons_col_sim_df)

    selected_columns = ['pt_score', 'ft_cls_score', 'ft_mlm_score', 'ft_cont_score', 'pt_sp_rank', 'ft_cls_rank', 'ft_mlm_sp_rank', 'ft_cont_sp_rank']

    # Compute statistics: min, max, mean
    stats = esm_cons_col_mean_sim_df[selected_columns].agg(['min', 'max', 'mean']).transpose()

    # Print the statistics
    print(f"Statistics for selected columns:\n{stats}")


    save_kde(esm_cons_col_sim_df, plot_dir, title='ESM2-3B', type='not-mean')
    save_kde(esm_cons_col_mean_sim_df, plot_dir, title='ESM2-3B', type='mean')

    save_kde_sprank(esm_cons_col_sim_df, plot_dir, title='ESM2-3B', type='not-mean')
    save_kde_sprank(esm_cons_col_mean_sim_df, plot_dir, title='ESM2-3B', type='mean')

    save_violin_plot(esm_cons_col_mean_sim_df, plot_dir, 'ESM2-3B')
    save_violin_sprank_plot(esm_cons_col_mean_sim_df, plot_dir, 'ESM2-3B')
    

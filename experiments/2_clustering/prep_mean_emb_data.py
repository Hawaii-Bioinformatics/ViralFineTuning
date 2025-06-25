import argparse
import pandas as pd
import torch


pkl_dir = 'pkl_dir'

def get_mean_rep(seq_id, emb_dir):
    try:
        seq_mean_rep = torch.mean(torch.load(f'{emb_dir}/{seq_id}.pt'), dim=1).cpu().detach().numpy()
        return seq_mean_rep
    except Exception as e:
        print('skipped ', seq_id)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, required=True, help="Path to directory with embeddings for the model")

    args = parser.parse_args()    

    emb_dir = args.emb_dir

    data = pd.read_pickle("vog_seq_test_df.pkl")
    # pdb.set_trace()
    data['emb'] = data['protein_id'].apply(lambda x: get_mean_rep(x, emb_dir))
    pkl_path = f"{pkl_dir}/esm_mean_emb.pkl"
    data.to_pickle(pkl_path)

    print('Done. Results saved to:', pkl_path)

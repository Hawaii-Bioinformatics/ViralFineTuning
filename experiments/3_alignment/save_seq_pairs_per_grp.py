import pandas as pd
from itertools import combinations


if __name__ == "__main__":
    df = pd.read_pickle("vog_seq_test_df.pkl")  # Modify as needed
    output_tsv = "all_pairs.tsv"
    with open(output_tsv, "w") as f:
        for group, gdf in df.groupby("group"):
            for (seq1_id1, seq1_str), (seq2_id, seq2_str) in combinations(gdf[['protein_id', 'protein_seq']].itertuples(index=False, name=None), 2):
                f.write(f"{seq1_id1}\t{seq2_id}\t{seq1_str}\t{seq2_str}\n")

    print('Done saving pairs at:', output_tsv)



import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import transformers
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from transformers import Trainer, DataCollatorForLanguageModeling, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from torch.distributed import barrier
from Bio import SeqIO
import inspect
import re
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm


# configs
msa_t_dim = 768
layer_esm2_3B = 36
layer_esm2_650M = 33
layer_esm2_150M = 30
layer_protTrans_xl = 24

# Single GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using gpu:', device)

torch.cuda.empty_cache()

print('loading mods')
chk_pt = "./checkpoint-1200"
tokenizer = T5Tokenizer.from_pretrained(chk_pt)
base_model = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = PeftModel.from_pretrained(base_model, chk_pt)
model = model.to(device)
print(model)
# read data
print('Reading data')

import pdb
pdb.set_trace()

# NOTE: for seq
dataset = pd.read_pickle(open("vog_seq_test_df.pkl", "rb"))
vog_seq_df = dataset.groupby('labels').filter(lambda x: len(x) >= 10)
print(vog_seq_df)
# ## NOTE: regen embs for seq len > 500 # TODO: remove for next time
# vog_seq_df = vog_seq_df[vog_seq_df.protein_seq.str.len()>=500]
# print(vog_seq_df)
seq_ids = vog_seq_df.protein_id.to_list()
sequences = vog_seq_df.protein_seq.to_list()
sequences = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in sequences]


output_dir = "prot_t5_xl/seq/ft_mlm_lora"
error_seqs = []
print('Generating embeddings...')

id_seq_zip = zip(seq_ids, sequences)
for i, (seq_id, sequence) in tqdm(enumerate(id_seq_zip), total=len(seq_ids)):
    encoded_tokens = tokenizer.batch_encode_plus([sequence], max_length=1024, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt') # padding=true -> longest in batch
    encoded_tokens["decoder_input_ids"] = encoded_tokens["input_ids"]
    encoded_tokens["labels"] = encoded_tokens["input_ids"]
    encoded_tokens = encoded_tokens.to(device)
    try:
        with torch.no_grad():
            outputs = model(**encoded_tokens, output_hidden_states=True)
        embedding = outputs.encoder_last_hidden_state
        torch.save(embedding[0], f'{output_dir}/{seq_id}.pt')
    except Exception as e:
        print(e)
        print('skipping seq_id:', seq_id)
        error_seqs.append(seq_id)
        continue
    
print('embeddings generated and saved')

    
with open('error_seqs.txt', 'w') as f:
    for seq in error_seqs:
        f.write(f"{seq}\n")   

print('Done')
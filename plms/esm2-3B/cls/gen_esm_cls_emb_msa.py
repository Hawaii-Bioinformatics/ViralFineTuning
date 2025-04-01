from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, Dataset
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EsmForSequenceClassification
#import evaluate
import torch
import pandas as pd
from tqdm import tqdm
import pdb

# directories
data_dir = "/data/rajan/vog"
pkl_dir = f"{data_dir}/data-pkls"
msa_fasta_dir = f"{data_dir}/fasta"
saved_mod_dir = "saved_mods"

emb_dir = f"{data_dir}/emb"
esm2_3B_emb_dir = f"{emb_dir}/esm2_3B"


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

# # NOTE: for MSA
dataset = pd.read_pickle(open("/data/rajan/vog/data-pkls/new_sel_msa_w_cons_col_df.pkl", "rb"))
print(dataset)
seq_ids = dataset.seq_id.to_list()
sequences = dataset.seq_str.to_list()
labels = set(dataset.align_id.tolist())
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}

print('loading mod')
chk_pt = "./checkpoint-7800"
tokenizer = AutoTokenizer.from_pretrained(chk_pt)
base_model = EsmForSequenceClassification.from_pretrained(
    "facebook/esm2_t36_3B_UR50D", num_labels=len(labels) , id2label=id2label, label2id=label2id)
print(base_model)
model = PeftModel.from_pretrained(base_model, chk_pt)
model = model.to(device)
print(model)

# NOTE: for seqs
# vog_seq_df['labels'] = vog_seq_df['labels'].astype('category')
# vog_seq_df['labels_codes'] = vog_seq_df['labels'].cat.codes
# labels = torch.tensor(vog_seq_df['labels_codes'].values, dtype = torch.long)

# NOTE: for msa
dataset['align_id'] = dataset['align_id'].astype('category')
dataset['labels_codes'] = dataset['align_id'].cat.codes
labels = torch.tensor(dataset['labels_codes'].values, dtype = torch.long)

error_seqs = []

output_dir = "/data/rajan/vog/emb/esm2_3B/msa/ft_cls_lora" # for msa

print('Generating embeddings...')

id_seq_zip = zip(seq_ids, sequences)
for i, (seq_id, sequence) in tqdm(enumerate(id_seq_zip), total=len(seq_ids)):
    # pdb.set_trace()
    encoded_tokens = tokenizer.batch_encode_plus(
        [sequence], max_length=1024, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt') # true = longest
    encoded_tokens["labels"] = torch.tensor(labels[i])
    encoded_tokens = encoded_tokens.to(device)
    try:
        with torch.no_grad():
            outputs = model(**encoded_tokens, output_hidden_states=True)
        embedding = outputs.hidden_states[layer_esm2_3B]
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
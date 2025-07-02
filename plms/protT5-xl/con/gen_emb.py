import pdb
import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.optim as optim
from transformers import T5Tokenizer, T5EncoderModel
from utils.model_zoo import AvgPoolingModel
from utils.lora_converter import convert_model
import re

def remap(layers):
    new_state_dict = {}
    for name, param in layers.items():
        new_name = ".".join(name.split(".")[1:])
        new_state_dict[new_name] = param

    return new_state_dict


def get_model_and_tokenizer(weight_path, device='cpu', model_type="t5"):
    base_model_path = "Rostlab/prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(base_model_path, legacy=False)
    print("tokenizer loaded")
    model = T5EncoderModel.from_pretrained(base_model_path)
    use_lora = True
    if use_lora:
        convert_model(model)

    state_dict = remap(torch.load(weight_path, map_location=device))
    model.load_state_dict(state_dict)
    return model, tokenizer


def run_model(model, tokenizer, sequence, device='cpu'):
    model.eval()
    with torch.no_grad():
        # tokenized_inputs = tokenizer([sequence])
        tokenized_inputs = tokenizer.batch_encode_plus([sequence], max_length=1024, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
        tokenized_inputs = torch.tensor(tokenized_inputs["input_ids"]).to(device)
        model = model.to(device)
        output = model(tokenized_inputs)
        output = output["last_hidden_state"]
        if output.is_cuda:
            output = output.cpu()
        output = output.numpy()
    return output

device = 'cuda'
t5_checkpoint = 'prott5_siamese_model_final_checkpoint_ep10.pt'

model, tokenizer = get_model_and_tokenizer(t5_checkpoint, device=device)


pdb.set_trace()

t5_ft_emb_dir = "prot_t5_xl/seq/contrastive"


# NOTE: for seq
dataset = pd.read_pickle(open("vog_seq_test_df.pkl", "rb"))
print(dataset)
filtered_df = dataset.groupby('labels').filter(lambda x: len(x) >= 10).copy()
seq_ids = filtered_df.protein_id.tolist()
sequences = filtered_df.protein_seq.to_list()
sequences = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in sequences]
print('len filtered_seq_ids:', len(seq_ids))


import pdb

pdb.set_trace()
id_seq_zip = zip(seq_ids, sequences)
for i, (seq_id, sequence) in tqdm(enumerate(id_seq_zip), total=len(seq_ids)):
    embedding = run_model(model, tokenizer, sequence, device=device)
    np.save(f"{t5_ft_emb_dir}/{seq_id}", embedding[0])
    # NOTE: embeddings have a start and end padding token
    torch.cuda.empty_cache()
print('Done. Embeddings saved at:', t5_ft_emb_dir)


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
from tokenizers import Tokenizer
import sys 

# Add the parent directory to sys.path
sys.path.append('/home/rsawhney/progen/progen2/')

from models.progen.modeling_progen import ProGenForCausalLM, ProGenForSequenceClassification


def remap(layers):
    new_state_dict = {}
    for name, param in layers.items():
        new_name = ".".join(name.split(".")[1:])
        new_state_dict[new_name] = param

    return new_state_dict


def get_model_and_tokenizer(weight_path, device='cpu', model_type="t5"):
    base_model_path = "/home/rsawhney/progen/progen2/checkpoints/progen2-large"
    tokenizer = create_progen_tokenizer_custom(file='/home/rsawhney/progen/progen2/tokenizer.json')
    tokenizer.enable_truncation(max_length=1024)
    print("tokenizer loaded")
    model = create_progen_model(base_model_path, type='causal')
    print(model)
    # model = AvgPoolingModel(model)
    # pdb.set_trace()
    use_lora = True
    if use_lora:
        convert_model(model)

    state_dict = remap(torch.load(weight_path, map_location=device))
    model.load_state_dict(state_dict)
    print(model)

    return model, tokenizer

### progen

def create_progen_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def create_progen_model(model_path, type = "classification"):
    if type == "causal" :
        return ProGenForCausalLM.from_pretrained(model_path)
    elif type == "classification" :
        return ProGenForSequenceClassification.from_pretrained(model_path)


def run_model(model, tokenizer, sequence, device='cpu'):
    model.eval()
    with torch.no_grad():
        tokenized_inputs = tokenizer.encode(sequence)
        tokenized_inputs = torch.tensor(tokenized_inputs.ids).to(device)
        model = model.to(device)
        output = model(tokenized_inputs, output_hidden_states=True)
        output = output["hidden_states"][-1]
        # if output.is_cuda:
        #     output = output.cpu()
        # output = output.numpy()
    return output

device = 'cuda'
progen_checkpoint = '/home/rsawhney/PooledAAEmbeddings/progen_siamese_model_final_checkpoint.pt'
model, tokenizer = get_model_and_tokenizer(progen_checkpoint, device=device)


# pdb.set_trace()

data_dir = "/data/rajan/vog"
emb_dir = f"{data_dir}/emb"
progen_ft_emb_dir = f"{emb_dir}/progen_large/seq/contrastive"
# output_path = '/data/rajan/vog/emb/esm2_3B/msa/contrastive'


# NOTE: for seq
dataset = pd.read_pickle(open("/data/rajan/vog/data-pkls/vog_seq_test_df.pkl", "rb"))
print(dataset)
filtered_df = dataset.groupby('labels').filter(lambda x: len(x) >= 10).copy()
seq_ids = filtered_df.protein_id.tolist()
sequences = filtered_df.protein_seq.to_list()
print('len filtered_seq_ids:', len(seq_ids))


# pdb.set_trace()
id_seq_zip = zip(seq_ids, sequences)
for i, (seq_id, sequence) in tqdm(enumerate(id_seq_zip), total=len(seq_ids)):
    embedding = run_model(model, tokenizer, sequence, device=device)
    np.save(f"{progen_ft_emb_dir}/{seq_id}", embedding)
    # NOTE: embeddings have a start and end padding token
    print('Embeddings saved for:', seq_id)
    torch.cuda.empty_cache()
print('Done. Embeddings saved at:', progen_ft_emb_dir)


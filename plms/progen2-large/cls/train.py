import os 
import torch
# from progen2.models.progen.modeling_progen import ProGenForCausalLM, ProGenForSequenceClassification
# from progen2.models.progen.configuration_progen import ProGenConfig
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
import datasets
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import seed_worker
from transformers import EsmForMaskedLM
import sys
import pdb

# Add the parent directory to sys.path
sys.path.append('progen/progen2/')

from models.progen.modeling_progen import ProGenForCausalLM, ProGenForSequenceClassification
from models.progen.configuration_progen import ProGenConfig

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["TORCH_USE_CUDA_DSA"]
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
args_fp16 = False
print(device)

# From github progen
def create_model(ckpt, fp16=True, type = "classification"):
    if fp16 and type == "causal" :
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    elif type == "causal" :
        return ProGenForCausalLM.from_pretrained(ckpt)
    if fp16 and type == "classification" :
        return ProGenForSequenceClassification.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    elif type == "classification" :
        return ProGenForSequenceClassification.from_pretrained(ckpt)
    
def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

# TOKENIZER
vocab_file = "progen/progen2/tokenizer.json" # from progen repo

tokenizer = create_tokenizer_custom(vocab_file)
# DATATA


class CustomDataset(Dataset):
    def __init__(self, tokenizer, input_ids, attention_mask, labels):
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # Préparer les labels, ici ils sont identiques aux input_ids pour le débruitage
        self.labels = labels

    def shift_tokens_right(self, input_ids, pad_token_id):
        """Décale les tokens vers la droite pour préparer decoder_input_ids."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        shifted_input_ids = input_ids.clone()
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = pad_token_id

        if input_ids.dim() == 1:
            shifted_input_ids = shifted_input_ids.squeeze(0)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        if self.attention_mask is not None : 
            attention_mask = self.attention_mask[idx]
        else : 
            attention_mask = None
        labels = self.labels[idx]

        if input_ids is None : 
            print(f"input_ids {idx} is None")

        decoder_input_ids = input_ids

        if self.attention_mask is not None : 

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels  
            }
        else : 
            return{

                "input_ids": input_ids,
                "labels": labels  
            }


train_dataset = pd.read_csv("train.tsv")
vog_min_number_seqs = 10
is_gt_than_min = train_dataset["#GroupName"].value_counts() >= vog_min_number_seqs
vogs_gt_min = train_dataset["#GroupName"].value_counts()[is_gt_than_min].index.unique()
cols_of_interest = ["#GroupName", "ProteinIDs", "sequence"]
dataset = train_dataset[train_dataset["#GroupName"].isin(vogs_gt_min)][cols_of_interest]
dataset = dataset[:100]

new_ids = dataset.ProteinIDs.str.split(".").apply(lambda x: ".".join(x[1:]))
dataset["ProteinIDs"] = new_ids

dataset.columns = ["labels", "protein_id", "protein_seq"]
train_df, test_df = train_test_split(dataset, test_size=0.1, stratify=dataset['labels'], random_state=1)
labels = set(dataset.labels.tolist())
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}

tokenizer.enable_truncation(max_length=1024)
tokenizer.enable_padding(length=1024)

# train_df = train_df.iloc[:10]
# test_df = test_df.iloc[:10]

train_ids = tokenizer.encode_batch(train_df['protein_seq'])
test_ids = tokenizer.encode_batch(test_df['protein_seq'])

train_list = [train_id.ids for train_id in train_ids]
train_mask_list = [train_id.attention_mask for train_id in train_ids]
test_list = [test_id.ids for test_id in test_ids]
test_mask_list = [test_id.attention_mask for test_id in test_ids]

input_ids = torch.tensor(train_list)
attention_mask = torch.tensor(train_mask_list)
train_df['labels'] = train_df['labels'].astype('category')
train_df['labels_codes'] = train_df['labels'].cat.codes
labels = torch.tensor(train_df['labels_codes'].values, dtype = torch.long)
num_labels = len(list(set(labels.tolist())))
ProGenConfig.num_labels = num_labels


test_input_ids = torch.tensor(test_list)
test_attention_mask = torch.tensor(test_mask_list)
test_df['labels'] = test_df['labels'].astype('category')
test_df['labels_codes'] = test_df['labels'].cat.codes
test_labels = torch.tensor(test_df['labels_codes'].values, dtype = torch.long)




def labels_to_one_hot(tensor, num_labels):
    # Crée un tenseur one-hot de taille (nombre d'exemples, num_labels)
    one_hot = torch.zeros(tensor.size(0), num_labels)
    # Utilise scatter_ pour remplir les positions correspondantes à chaque label
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot

labels = labels_to_one_hot(labels, num_labels)
test_labels = labels_to_one_hot(test_labels, num_labels)

# MODEL

ckpt = "progen/progen2/checkpoints/progen2-large/" # from their repo

model = create_model(ckpt=ckpt, fp16=args_fp16, type = "classification")
#print(model)

for name, param in model.named_parameters():
        param.requires_grad = False

loraft = True
if loraft : 
    
    targets = ["qkv_proj","out_proj"] # ["qkv_proj", "out_proj", "fc_in", "fc_out"] for plm lora ft
    lora_rank = 8
    lora_alpha = 8
    lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="loftq",
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            layers_to_transform = [i for i in range(32) if i % 1 == 0])
    
    model = get_peft_model(model, lora_config).to(device)
    print(model)


lora_params = []
for name, param in model.named_parameters():
    if "lora" in name : 
        param.requires_grad = True
        lora_params.append(param)

pdb.set_trace()

clm = False
if clm : 
    labels, test_labels = input_ids, test_input_ids
    
train_dataset = CustomDataset(tokenizer= tokenizer, input_ids=input_ids, attention_mask=attention_mask ,labels = labels)
test_dataset = CustomDataset(tokenizer= tokenizer, input_ids=test_input_ids, attention_mask=test_attention_mask ,labels = test_labels)
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
train_list = [data for data in data_loader]
print(train_list[0])
test_list = [data for data in test_dataloader]

import pdb
pdb.set_trace()

training_args = transformers.TrainingArguments(
    output_dir="./results-progen-mlm",
    evaluation_strategy="epoch",
    do_eval=True,
    dataloader_pin_memory=False,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=1.0,
    weight_decay=0.01)

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

print('Running trainer')

# Trainer modified : 
#      l.838 : in get_train_dataloader() : comment column removing + create custom collator and right it dataloader params. 
#      l.2130 : removed try / except used for LongVirus

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_list,
    eval_dataset=test_list,
    data_collator=data_collator,
    tokenizer=tokenizer)

#ckpt2 = "./results-progen-cls/checkpoint-6750"
#trainer.train(resume_from_checkpoint=ckpt2)

trainer.train()
print('Model trained')

out_name = "progen_vog_ft"
trainer.save_model(out_name)
model.base_model.save_pretrained(save_directory = "./results-progen-mlm", safe_serialization = False)
print(f'Saved trainer as {out_name} in ./results-progen-mlm')

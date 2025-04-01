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
from transformers import T5EncoderModel, T5Tokenizer
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from Bio import SeqIO
import inspect
import re
import pandas as pd
from torch.utils.data import DataLoader
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Nombre de GPU disponibles
num_gpus = torch.cuda.device_count()

print(f"Nombre de GPU disponibles : {num_gpus}")

# Liste chaque GPU disponible et son nom
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def load_fasta_as_tuples(fasta_path):
    sequences = []
    # Parcourir chaque enregistrement du fichier FASTA
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Ajouter un tuple (identifiant, séquence) à la liste
        sequences.append((record.id, record.seq))
    return sequences

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Rostlab/prot_t5_xl_uniref50")
    model_type: Optional[str] = field(default="prot_t5_xl_uniref50")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default= None,
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    output_dir: str = field(
        default="/home/rsawhney/models/clm-protrans-lora/new", 
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    save_steps: float = field(
        default=1000,
        metadata={"help": "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."},
    )
    logging_steps: float = field(
        default=200,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    gradient_accumulation_steps: int = field(
        default=128,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    dataloader_pin_memory: bool = field(
        default=False, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})

    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )    

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenizer, input_ids, attention_mask):
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # Préparer les labels, ici ils sont identiques aux input_ids pour le débruitage
        self.labels = input_ids.clone()

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
        # Obtenir input_ids et attention_mask pour un index spécifique
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]

        if input_ids is None : 
            print(f"input_ids {idx} is None")

        # print(input_ids)
        # print(attention_mask)
        # print(labels)

        #if input_ids.dim() == 1:
        #    input_ids = input_ids.unsqueeze(0)
        #if labels.dim() == 1:
        #    labels = labels.unsqueeze(0)

        # Préparer les decoder_input_ids
        decoder_input_ids = input_ids #self.shift_tokens_right(input_ids, self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels  # assurez-vous que les labels sont corrects pour votre cas d'utilisation
        }



def train():

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = transformers.T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    # train_dataset = pd.read_csv("~/vog_train_dataset.tsv")
    train_dataset = pd.read_csv('/home/rsawhney/finetune_esm_classification/train_dataset.tsv')

    # Keep only vogs with 10 sequences since we classify back to the vog

    vog_min_number_seqs = 10
    is_gt_than_min = train_dataset["#GroupName"].value_counts() >= vog_min_number_seqs
    vogs_gt_min = train_dataset["#GroupName"].value_counts()[is_gt_than_min].index.unique()
    cols_of_interest = ["#GroupName", "ProteinIDs", "sequence"]
    dataset = train_dataset[train_dataset["#GroupName"].isin(vogs_gt_min)][cols_of_interest]
    dataset = dataset[:100] # TODO: remove
    new_ids = dataset.ProteinIDs.str.split(".").apply(lambda x: ".".join(x[1:]))
    dataset["ProteinIDs"] = new_ids

    dataset.columns = ["labels", "protein_id", "protein_seq"]
    # train_df, test_df = train_test_split(dataset, test_size=0.05)
    train_df, test_df = train_test_split(dataset, test_size=0.1, stratify=dataset['labels'], random_state=1)
    #print(train_df)
    
    train_df["protein_seq"] = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in train_df["protein_seq"]]
    test_df["protein_seq"] = [" ".join(list(re.sub(r"[UZOB-]", "X", sequence))) for sequence in test_df["protein_seq"]]
    print(train_df)

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(train_df['protein_seq'], max_length = 512, add_special_tokens=True, padding="max_length",truncation=True, return_tensors='pt')
    test_ids = tokenizer.batch_encode_plus(test_df['protein_seq'], max_length = 512, add_special_tokens=True, padding="max_length",truncation=True, return_tensors='pt')

    print(ids.keys())

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # torch.save(ids['input_ids'], '/home/thibaut/protrans/tokenized_input_ids.pt')
    # torch.save(ids['attention_mask'], '/home/thibaut/protrans/tokenized_attention_mask.pt')

    test_input_ids = torch.tensor(test_ids['input_ids']).to(device)
    test_attention_mask = torch.tensor(test_ids['attention_mask']).to(device)

    # torch.save(test_input_ids, '/home/thibaut/protrans/test_tokenized_input_ids.pt')
    # torch.save(test_attention_mask, '/home/thibaut/protrans/test_tokenized_attention_mask.pt')


    # input_ids = torch.load('/scratch/bbtw/mbelcaid/thibaut/protrans/tokenized_input_ids.pt')
    # attention_mask = torch.load('/scratch/bbtw/mbelcaid/thibaut/protrans/tokenized_attention_mask.pt')
    # test_input_ids = torch.load('/scratch/bbtw/mbelcaid/thibaut/protrans/test_tokenized_input_ids.pt')
    # test_attention_mask = torch.load('/scratch/bbtw/mbelcaid/thibaut/protrans/test_tokenized_attention_mask.pt')

    # Assurez-vous de les envoyer sur le bon périphérique si vous utilisez CUDA
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    test_input_ids = test_input_ids.to(device)
    test_attention_mask = test_attention_mask.to(device)

    print(input_ids.shape)
    print(attention_mask.shape)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataset = CustomDataset(tokenizer= tokenizer, input_ids=input_ids, attention_mask=attention_mask)
    test_dataset = CustomDataset(tokenizer= tokenizer, input_ids=test_input_ids, attention_mask=test_attention_mask)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    """

    train_df, test_df = train_test_split(dataset, test_size=0.05)

    labels = set(dataset.labels.tolist())
    id2label = {i:l for i,l in enumerate(labels)}
    label2id = {l:i for i,l in enumerate(labels)}

    train_df['protein_seq'] = train_df['protein_seq'].apply(lambda x : " ".join(list(re.sub(r"[UZOB-]", "X", x))))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trn = Dataset.from_dict({'labels':train_df.labels,'protein_id':train_df.protein_id, 'protein_seq': train_df.protein_seq})
    tst = Dataset.from_dict({'labels':test_df.labels,'protein_id':test_df.protein_id, 'protein_seq': test_df.protein_seq})
    dataset_dict = DatasetDict({'train': trn, 'validation': tst})
    
    def tokenize_and_format(dataset):
        # This should return PyTorch tensors directly
        tokenized_outputs = tokenizer.batch_encode_plus(
            dataset['protein_seq'],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors='pt')
        
        # Convert labels to tensors directly here
        tokenized_outputs['labels'] = torch.tensor([label2id[x] for x in dataset['labels']])
        return tokenized_outputs
    
    def convert_to_tensors(batch):
        # Ensure that input_ids and attention_mask are converted to tensors with the correct shape
        batch['input_ids'] = torch.tensor(batch['input_ids'])
        batch['attention_mask'] = torch.tensor(batch['attention_mask'])
        batch['labels'] = torch.tensor(batch['labels'])
        return batch
    
    tokenized_dataset = dataset_dict.map(tokenize_and_format, batched=True, batch_size=50)
    tokenized_dataset = tokenized_dataset.map(convert_to_tensors, batched=True, batch_size=50)
    
    tokenized_dataset.save_to_disk("/home/thibaut/protrans/")
    #tokenized_dataset = load_from_disk("/home/thibaut/protrans/")

    print(type(tokenized_dataset["train"]['labels']))
    print(type(tokenized_dataset["train"]['input_ids']))
    print(type(tokenized_dataset["train"]['input_ids'][0]))
    """
    targets = ["q","k","v","o"]
    lora_rank = 8
    lora_alpha = 8

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=targets,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform = [i for i in range(25) if i % 1 == 0], 
    )

    # print(lora_config)

    model = get_peft_model(model, lora_config).to(device)
    # grad_output contient les gradients par rapport aux sorties du module

    # Gel des poids des couches du modèles
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Dégel des couches que l'on veut entrainer. 
    if training_args.trainable_params is not None:
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(param)
            param.requires_grad = True
    pdb.set_trace()
    print_trainable_parameters(model)
    pdb.set_trace()

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset = train_dataset,
        #eval_dataset = test_dataset,
        data_collator= data_collator)
    
    trainer.train()
    print('trained ! ')
    trainer.save_state()

    model.base_model.save_pretrained(save_directory = training_args.output_dir, safe_serialization = False)
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    print('start')
    train()
    

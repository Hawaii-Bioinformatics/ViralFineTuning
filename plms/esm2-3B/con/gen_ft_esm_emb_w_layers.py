'''
Script to generate embeddings using ESM2-3B finetuned with siamese network

To run:
    python gen_finetuned_esm_emb.py [fasta_file_path] [output_path] [layers]
    Example: python gen_finetuned_esm_emb.py temp/test.fasta temp/ 2,4,30,36
'''

import pdb

import argparse
from Bio import SeqIO
import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


device = 'cuda'


def get_model_and_tokenizer(weight_path, device='cpu', model_type=esm.pretrained.esm2_t36_3B_UR50D):
    model, alphabet = model_type()
    tokenizer = alphabet.get_batch_converter()
    state_dict = remap(torch.load(weight_path, map_location=device))
    model.load_state_dict(state_dict)
    return model, tokenizer


def remap(layers):
    new_state_dict = {}
    for name, param in layers.items():
        new_name = ".".join(name.split(".")[1:])
        new_state_dict[new_name] = param

    return new_state_dict


def run_model(model, tokenizer, sequence, layers=None, device='cpu'):
    if layers==None:
        layers = model.num_layers # last layer

    model.eval()
    with torch.no_grad():
        # pdb.set_trace()
        _, _, tokenized_inputs = tokenizer([sequence])
        tokenized_inputs = tokenized_inputs.to(device)
        model = model.to(device)
        output = {}
        output["mean_representations"] = {}
        output["representations"] = model(tokenized_inputs, repr_layers=layers)["representations"]
        # remove padding bos, eos
        for layer in layers:
            output["representations"][layer] = output["representations"][layer][:,1:-1,:] 
            output["mean_representations"][layer] = torch.mean(output["representations"][layer], dim=1)[0]
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Process sequences from a FASTA file and save embeddings.")
    parser.add_argument('fasta_file_path', type=str, help='Path to the input FASTA file')
    parser.add_argument('output_path', type=str, help='Path to save the output embeddings')
    parser.add_argument('layers', type=str, help='Comma-separated model layers')
    return parser.parse_args()


def main():
    args = parse_args()
    fasta_file_path = args.fasta_file_path
    output_path = args.output_path
    layers = [int(layer) for layer in args.layers.split(',')]

    print('Fasta file = ', fasta_file_path)
    records = list(SeqIO.parse(fasta_file_path, "fasta"))
    num_records = len(records)  # number of sequences
    print('Number of sequences = ', num_records)
    print('\nLayers = ', layers)

    print('Loading model...')
    model, tokenizer = get_model_and_tokenizer('3b_model_checkpoint.pt', device=device)
    print('Model loaded')

    # Process each sequence and save the embeddings
    for record in tqdm(records, total=num_records, desc="Processing sequences"):
        seq_id = record.id
        sequence = str(record.seq)
        embedding = run_model(model, tokenizer, (seq_id, sequence), layers, device=device)
        assert(len(sequence) == embedding["representations"][layers[1]].size()[1])
        # Save the embedding
        torch.save(embedding, f"{output_path}/{seq_id}.pt")

    print('Done. Embeddings saved at:', output_path)


if __name__ == "__main__":
    main()
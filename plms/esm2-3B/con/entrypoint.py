import pdb

import argparse
import sys
import os
from process_data import create_dataset_folders
from run_training import build_and_train_model, MODEL_TYPES, protocolS, resume_training_run
from utils.model_zoo import BASE_MODELS, build_avg_pooling_model, build_test_model
from test import evaluate_test_data
import torch
import datetime

if __name__ == "__main__":
    print("Let's go")
    parser = argparse.ArgumentParser(description="Process data from a specified path and optionally create a train-test split.")
    parser.add_argument("--base-model", type=str, default=BASE_MODELS[0], help=f"base esm model to use, options are {BASE_MODELS}")
    parser.add_argument("--model-type", type=str, default='avg_pooling', help=f"model type, must be one of {MODEL_TYPES}")
    parser.add_argument("--data-path", type=str, default='tmp', help="Path to the data directory")
    parser.add_argument("--protocol", help=f"protocol to train with, should be one of {protocolS}")
    parser.add_argument("--learning-rate", type=float, default=0.000002, help="learning rate for training (default: 0.000002)")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--validation-size", type=int, default=10, help="number of validation samples to run after each epoch")
    parser.add_argument("--train-size", type=int, default=None, help="number of train samples to run during each epoch")
    parser.add_argument("--max-length", type=int, default=1024, help="max length of amino acid sequence to load")
    parser.add_argument("--use-lora", action='store_true', help="to use LoRA (https://arxiv.org/pdf/2106.09685.pdf) of not")
    parser.add_argument("--load-checkpoint", type=str, default='none_given', help="load checkpoint")
    parser.add_argument("--early-stopping", action='store_true', help="early stopping during training")
    parser.add_argument("--positive-count", type=int, default=6, help="number of positive samples for sup mpn")
    parser.add_argument("--negative-count", type=int, default=4, help="number of negative samples for sup mpn")
    parser.add_argument("--model-parallelism", action="store_true", help="use model parallelism")
    parser.add_argument("--data-parallelism", action="store_true", help="use model parallelism")
    parser.add_argument("--device", type=str, default='none_given', help="device")
    parser.add_argument("--embedding-path", type=str, default='none_given', help="path to load embeddings of amino acid sequences when training")
    parser.add_argument("--train-path", help="Path for train data")
    parser.add_argument("--test-path", help="Path for test data")
    parser.add_argument("--create-train-test-split", action="store_true", help="Create a train/test split")
    parser.add_argument("--train", action="store_true", help="Create a train/test split")
    parser.add_argument("--test", action="store_true", help="Calculate spearman correlation")
    parser.add_argument("--test-checkpoint", help="Path to load model checkpoint for test")
    parser.add_argument("--split-ratio", type=float, default=0.15, help="Ratio for train-test split (default: 0.1)")
    parser.add_argument("--cosine-annealing", action="store_true", help="use cosine annealing")
    parser.add_argument("--timed_checkpoint_path", type=str, default='none_given', help="path to dump checkpoints")
    parser.add_argument("--checkpoint_frequency", type=str, default='none_given', help="frequency")
    parser.add_argument("--resume", action="store_true", help="resume run")
    parser.add_argument("--resume-checkpoint", help="Path to load checkpoint")
    parser.add_argument("--current-epoch", help="manually enter current epoch", default=0, type=int)
    args = parser.parse_args()
    print(args)
    for arg_name in vars(args):
        if getattr(args, arg_name) == 'none_given':
            setattr(args, arg_name, None)

    base_model = args.base_model
    data_path = args.data_path
    train_path = args.train_path
    test_path = args.test_path
    create_split = args.create_train_test_split
    train = args.train
    test = args.test
    split_ratio = args.split_ratio
    protocol = args.protocol

    model_type=args.model_type
    test_checkpoint=args.test_checkpoint
    epochs=args.epochs
    batch_size=args.batch_size
    learning_rate=args.learning_rate
    validation_size=args.validation_size
    train_size=args.train_size
    max_length=args.max_length
    use_lora=args.use_lora
    load_checkpoint=args.load_checkpoint
    early_stopping=args.early_stopping
    embedding_path=args.embedding_path
    positive_count=args.positive_count
    negative_count=args.negative_count
    model_parallel=args.model_parallelism
    data_parallel=args.data_parallelism
    device=args.device
    cosine_annealing=args.cosine_annealing
    timed_checkpoint_path=args.timed_checkpoint_path
    checkpoint_frequency=int(args.checkpoint_frequency) if args.checkpoint_frequency else args.checkpoint_frequency
    resume=args.resume
    resume_checkpoint=args.resume_checkpoint
    current_epoch=args.current_epoch

    if resume:
        resume_training_run(resume_checkpoint)

    if create_split:
        create_dataset_folders(data_path, train_path, split_ratio)

    if train:
        # pdb.set_trace()
        #create directory to store run
        if not os.path.exists('training_runs'):
            os.makedirs('training_runs')
            
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f'training_runs/training_run_{current_datetime}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        #save arguments for run
        output_file_path = os.path.join(dir_name, 'arguments.txt')
        with open(output_file_path, 'w') as file:
            file.write(f"Arguments and Values for run at {current_datetime}:\n")
            for arg, value in vars(args).items():
                file.write(f"{arg}: {value}\n")

        save_after_epoch_path = os.path.join(dir_name)
        build_and_train_model(base_model=base_model,
                        model_type=model_type, 
                        protocol=protocol, 
                        data_path=data_path, 
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        validation_size=validation_size,
                        train_size=train_size,
                        max_length=max_length,
                        use_lora=use_lora,
                        save_after_epoch_path=save_after_epoch_path,
                        load_checkpoint=load_checkpoint,
                        early_stopping=early_stopping,
                        embedding_path=embedding_path,
                        positive_count=positive_count,
                        negative_count=negative_count,
                        model_parallel=model_parallel,
                        data_parallel=data_parallel,
                        device=device,
                        cosine_annealing=cosine_annealing,
                        timed_checkpoint_path=timed_checkpoint_path,
                        checkpoint_frequency=checkpoint_frequency,
                        current_epoch=current_epoch)
    # TODO: undo false below
    test = False
    if test:
        if model_type == 'avg_pooling':
            model, tokenizer = build_avg_pooling_model(base_model)
        if model_type == 'test':
            model, tokenizer = build_test_model(base_model, embedding=True)
        if test_checkpoint:
            dict = torch.load(test_checkpoint)
            model.load_state_dict(dict)
        evaluate_test_data(model, tokenizer, test_path)


import pdb
import torch
import boto3
import datetime
import torch.utils.data as data_utils
from utils.model_zoo import build_avg_pooling_model
import pickle
import zipfile
import os
from botocore.exceptions import NoCredentialsError



def save_to_s3(local_directory, s3_bucket, s3_prefix=''):
    """
    Zip an entire local directory and upload the zip archive to an S3 bucket.

    Args:
        local_directory (str): The local directory to zip and upload.
        s3_bucket (str): The name of the S3 bucket.
        s3_prefix (str, optional): The S3 prefix (folder) under which to store the zip archive.

    Returns:
        bool: True if the directory was zipped and uploaded successfully.
    """
    try:
        # Create a temporary zip file
        zip_filename = 'temp.zip'
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(local_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, local_directory))

        # Upload the zip file to S3
        s3_client = boto3.client('s3',
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY
        )
        s3_object_key = os.path.join(s3_prefix, os.path.basename(zip_filename))
        s3_client.upload_file(zip_filename, s3_bucket, s3_object_key)

        # Clean up the temporary zip file
        os.remove(zip_filename)

        return True

    except Exception as e:
        print(f"An error occurred while zipping and uploading the directory to S3: {e}")
        return False

def save_model(model, filepath, s3_bucket=None, s3_key=None):
    """
    Save a PyTorch model to a specified file path and optionally upload it to an S3 bucket.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        filepath (str): The file path where the model will be saved locally.
        s3_bucket (str, optional): The name of the S3 bucket to upload the model to.
        s3_key (str, optional): The S3 object key (file path) under which the model will be stored.

    Returns:
        bool: True if the model was saved successfully.
    """
    try:
        # Save the model locally
        torch.save(model.state_dict(), filepath)
        return True

    except Exception as e:
        print(f"An error occurred while saving/uploading the model: {e}")
        return False

def save_data_loader(dataloader, filepath):
    """
    Save a DataLoader object as a pickle object to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader to be saved.
        filepath (str): The file path where the DataLoader will be saved as a pickle object.

    Returns:
        bool: True if the DataLoader was saved successfully.
    """
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(dataloader, file)

        return True

    except Exception as e:
        print(f"An error occurred while saving the DataLoader: {e}")
        return False

def save_optimizer(optimizer, filepath, s3_bucket=None, s3_key=None):
    """
    Save a PyTorch optimizer to a specified file path and optionally upload it to an S3 bucket.

    Args:
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to save.
        filepath (str): The file path where the optimizer will be saved locally.
        s3_bucket (str, optional): The name of the S3 bucket to upload the optimizer to.
        s3_key (str, optional): The S3 object key (file path) under which the optimizer will be stored.

    Returns:
        bool: True if the optimizer was saved successfully.
    """
    try:
        # Save the optimizer locally
        torch.save(optimizer.state_dict(), filepath)
        return True

    except Exception as e:
        print(f"An error occurred while saving/uploading the optimizer: {e}")
        return False

def save_scheduler(scheduler, filepath, s3_bucket=None, s3_key=None):
    """
    Save a PyTorch scheduler to a specified file path and optionally upload it to an S3 bucket.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to save.
        filepath (str): The file path where the scheduler will be saved locally.
        s3_bucket (str, optional): The name of the S3 bucket to upload the scheduler to.
        s3_key (str, optional): The S3 object key (file path) under which the scheduler will be stored.

    Returns:
        bool: True if the scheduler was saved successfully.
    """
    try:
        # Save the scheduler locally
        torch.save(scheduler.state_dict(), filepath)
        return True

    except Exception as e:
        print(f"An error occurred while saving/uploading the scheduler: {e}")
        return False

def full_save(model=None, optimizer=None, scheduler=None, data_loader=None, directory=None, bucket=None, current_epoch=None, current_iteration=None):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(os.path.join(directory, 'temp_checkpoints')):
        os.mkdir(os.path.join(directory, 'temp_checkpoints'))
    # Create a dictionary to hold all checkpoint data
    checkpoint = {
        'model_state_dict': model.state_dict() if model else None,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'current_epoch': current_epoch,
        'current_iteration': current_iteration,
    }
    # Save model, optimizer, scheduler, epoch, iteration in one checkpoint file
    model_path = os.path.join(directory, 'temp_checkpoints', f'training_checkpoint_{current_datetime}.pt')
    torch.save(checkpoint, model_path)
    if data_loader:
        data_loader_path = os.path.join(directory, 'temp_checkpoints', f'data_loader_temp_checkpoint{current_datetime}.pkl')
        save_data_loader(data_loader, data_loader_path)
        
    # if model:
    #     model_path = os.path.join(directory, 'temp_checkpoints', f'model_temp_checkpoint{current_datetime}.pt')
    #     save_model(model, model_path)
    # if scheduler:
    #     scheduler_path = os.path.join(directory, 'temp_checkpoints', f'scheduler_temp_checkpoint{current_datetime}.pt')
    #     save_scheduler(scheduler, scheduler_path)
    # if optimizer:
    #     optimizer_path = os.path.join(directory, 'temp_checkpoints', f'optimizer_temp_checkpoint{current_datetime}.pt')
    #     save_optimizer(optimizer, optimizer_path)


    # if bucket:
    #     save_to_s3(directory, bucket, directory)
    

def load_model_optimizer_scheduler(directory, model_class, scheduler_class, optimizer_class, device):
    """
    Load a PyTorch model, optimizer, and scheduler from the latest checkpoint in a directory.

    Args:
        directory (str): The directory where the checkpoints are saved.
        model_class (nn.Module): The model class instance (e.g., MyModelClass()).
        optimizer_class (Optimizer): The optimizer class instance (e.g., torch.optim.Adam()).
        scheduler_class (Scheduler): The scheduler class instance (e.g., torch.optim.lr_scheduler.StepLR()).
        device (torch.device): The device to map the model and optimizer to (e.g., torch.device('cuda') or 'cpu').

    Returns:
        tuple: A tuple containing the loaded model, optimizer, scheduler, current_epoch, current_iteration.
    """
    try:
        # model_checkpoint = max(
        #     (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'model_temp_checkpoint' in file),
        #     key=os.path.getctime
        # )
        # print('model_checkpoint: ', model_checkpoint)
        # optimizer_checkpoint = max(
        #     (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'optimizer_temp_checkpoint' in file),
        #     key=os.path.getctime
        # )
        # print('optimizer_checkpoint: ', optimizer_checkpoint)

        # scheduler_checkpoint = max(
        #     (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'scheduler_temp_checkpoint' in file),
        #     key=os.path.getctime
        # )

        # model_class.load_state_dict(torch.load(model_checkpoint, map_location=device))
        # optimizer_class.load_state_dict(torch.load(optimizer_checkpoint, map_location=device))
        # scheduler_class.load_state_dict(torch.load(scheduler_checkpoint))

        # pdb.set_trace()
        # Find the latest unified checkpoint file
        latest_checkpoint = max(
            (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'training_checkpoint_' in file),
            key=os.path.getctime
        )
        print('Loading latest checkpoint: ', latest_checkpoint)

        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        # Load model state
        if 'model_state_dict' in checkpoint:
            model_class.load_state_dict(checkpoint['model_state_dict'])
            # model_class = model_class.to(device)
            print('Model state loaded.')

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer_class.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Optimizer state loaded.')

        # # Load scheduler state
        # if 'scheduler_state_dict' in checkpoint and scheduler_class:
        #     scheduler_class.load_state_dict(checkpoint['scheduler_state_dict'])
        #     print('Scheduler state loaded.')

        # Ensure the optimizer's state is moved to the correct device
        def ensure_optimizer_device(optimizer, device):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        ensure_optimizer_device(optimizer_class, device)

        # Optionally, verify the optimizer's device
        # def check_optimizer_device(optimizer):
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 print(f'Tensor {k} is on device {v.device}')
        # check_optimizer_device(optimizer_class)
        # pdb.set_trace()

        # # Load current epoch and iteration
        # current_epoch = checkpoint.get('current_epoch', 0) + 1
        current_epoch = 9
        # current_iteration = 0 # NOTE: override
        # current_iteration = checkpoint.get('current_iteration', 0)
        current_iteration = 30000
        print(f'Resuming from epoch {current_epoch}, iteration {current_iteration}')

        return model_class, optimizer_class, scheduler_class, current_epoch, current_iteration

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None, 0, 0  # Return default values if loading fails



## NOTE: original function modified by the one above
# def load_model_optimizer_scheduler(directory, model_class, scheduler_class, optimizer_class, device):
#     # pdb.set_trace()
#     """
#     Load a PyTorch model, optimizer, and scheduler from saved checkpoints in a directory.

#     Args:
#         directory (str): The directory where the checkpoints are saved.
#         model_class (type): The class/type of the model (e.g., MyModelClass).
#         optimizer_class (type): The class/type of the optimizer (e.g., MyOptimizerClass).
#         scheduler_class (type): The class/type of the scheduler (e.g., MySchedulerClass).

#     Returns:
#         tuple: A tuple containing the loaded model, optimizer, and scheduler.
#     """
#     try:
#         # Find the latest checkpoint files
#         model_checkpoint = max(
#             (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'model_temp_checkpoint' in file),
#             key=os.path.getctime
#         )
#         print('model_checkpoint: ', model_checkpoint)
#         optimizer_checkpoint = max(
#             (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'optimizer_temp_checkpoint' in file),
#             key=os.path.getctime
#         )
#         print('optimizer_checkpoint: ', optimizer_checkpoint)

#         # scheduler_checkpoint = max(
#         #     (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'scheduler_temp_checkpoint' in file),
#         #     key=os.path.getctime
#         # )

#         model_class.load_state_dict(torch.load(model_checkpoint, map_location=device))
#         optimizer_class.load_state_dict(torch.load(optimizer_checkpoint, map_location=device))
#         # scheduler_class.load_state_dict(torch.load(scheduler_checkpoint))

#         def ensure_optimizer_device(optimizer, device):
#             for state in optimizer.state.values():
#                 for k, v in state.items():
#                     if isinstance(v, torch.Tensor):
#                         state[k] = v.to(device)
#         ensure_optimizer_device(optimizer_class, device)

#         # Optionally, verify the devices
#         def check_optimizer_device(optimizer):
#             for state in optimizer.state.values():
#                 for k, v in state.items():
#                     if isinstance(v, torch.Tensor):
#                         print(f'Tensor {k} is on device {v.device}')
        
#         # check_optimizer_device(optimizer_class)
                        
#         return model_class, optimizer_class

#     except Exception as e:
#         print(f"An error occurred while loading the model, optimizer, and scheduler: {e}")
#         return None, None, None

def load_data_loader_from_pickle(directory):
    """
    Load a DataLoader object from a pickle file on disk.

    Args:
        filepath (str): The file path of the saved DataLoader pickle object.

    Returns:
        torch.utils.data.DataLoader: The loaded DataLoader object.
    """
    try:
        dataloader_checkpoint = max(
            (os.path.join(directory, 'temp_checkpoints', file) for file in os.listdir(os.path.join(directory, 'temp_checkpoints')) if 'data_loader_temp_checkpoint' in file),
            key=os.path.getctime
        )
        print('dataloader_checkpoint = ', dataloader_checkpoint)
        with open(dataloader_checkpoint, 'rb') as file:
            loaded_dataloader = pickle.load(file)
        return loaded_dataloader

    except Exception as e:
        print(f"An error occurred while loading the DataLoader: {e}")
        return None
    
type_map = {
    "train_size": int,

}

argparse_defaults = {
    "base_model": "esm2_t30_150M_UR50D",
    "model_type": "avg_pooling",
    "data_path": "tmp",
    "protocol": "None",  # Replace with the actual default value
    "learning_rate": 0.000002,
    "epochs": 2,
    "batch_size": 1,
    "validation_size": 1000000,
    "train_size": None,  # Replace with the actual default value
    "max_length": 1200,
    "use_lora": False,
    "load_checkpoint": None,
    "early_stopping": False,
    "positive_count": 6,
    "negative_count": 4,
    "model_parallelism": False,
    "data_parallelism": False,
    "device": None,
    "embedding_path": None,
    "train_path": None,  # Replace with the actual default value
    "test_path": None,  # Replace with the actual default value
    "create_train_test_split": False,
    "train": True,
    "test": False,
    "test_checkpoint": None,  # Replace with the actual default value
    "split_ratio": 0.15,
    "cosine_annealing": True,
    "timed_checkpoint_path": None,
    "save_after_epoch_path": None,
    "checkpoint_frequency": None,  # Replace with the actual default value
}

def parse_config_file(config_file):
    """
    Parse a configuration file and return a dictionary of parsed variables, with default values
    for unspecified variables based on argparse_defaults.

    Args:
        config_file (str): Path to the configuration file.
        argparse_defaults (dict): Dictionary containing argparse default values.

    Returns:
        dict: A dictionary of parsed variables.
    """
    # Initialize a dictionary with argparse default values
    parsed_variables = argparse_defaults.copy()

    # Read the content of the configuration file
    with open(config_file, 'r') as file:
        lines = [_ for _ in file][1:]
        for line in lines:
            line = line.strip()
            print(line)
            if line:
                key, value = line.split(": ")
                key = key.strip()
                value = value.strip()
                print(key)
                print(value)
                # Handle 'None' values as is
                if value.lower() == 'none':
                    parsed_variables[key] = None
                else:
                    # Convert the value to the appropriate type if needed
                    if key in argparse_defaults:
                        if isinstance(argparse_defaults[key], bool):
                            value = value.lower() == "true"
                        elif isinstance(argparse_defaults[key], int):
                            try:
                                value = int(value)
                            except ValueError:
                                pass
                        elif isinstance(argparse_defaults[key], float):
                            try:
                                value = float(value)
                            except ValueError:
                                pass

                    elif key in type_map:
                        _type = type_map[key]
                        value = _type(value)

                    parsed_variables[key] = value

    return parsed_variables

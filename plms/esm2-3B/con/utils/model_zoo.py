import pdb
import torch
import torch.nn as nn
from typing import List, Optional
import esm
import ankh
import re
from .esm2 import ESM2
from .custom_t5 import CustomT5Model
from torchvision import models
import sys
import os
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

from transformers import (AutoTokenizer, AutoModel, AutoConfig, T5Tokenizer, T5ForConditionalGeneration, T5ForSequenceClassification,
     T5EncoderModel, EsmForMaskedLM, EsmForSequenceClassification)

from tokenizers import Tokenizer

# Add the parent directory to sys.path
sys.path.append('/home/rsawhney/progen/progen2/')

from models.progen.modeling_progen import ProGenForCausalLM, ProGenForSequenceClassification

BASE_MODELS = ['esm2_t36_3B_UR50D', 'Rostlab/prot_t5_xl_uniref50', '']

def load_data(base_model):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{base_model}.pt"
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data

def build_base_esm_tokenizer(base_model):
    _, alphabet = getattr(esm.pretrained, base_model)()
    tokenizer = alphabet.get_batch_converter()
    def tokenizer_function(inputs):
        assert isinstance(inputs, list), "pass in list"
        formatted_inputs = []
        for i, input in enumerate(inputs):
            formatted_inputs.append((f'protien{i}', input))
        _,_,output = tokenizer(formatted_inputs)
        return output
    return tokenizer_function

def build_base_ankh_tokenizer(_tokenizer):
    def tokenizer_function(inputs):
        assert isinstance(inputs, list), "pass in list"
        protein_sequences = [list(seq) for seq in inputs]
        outputs = _tokenizer.batch_encode_plus(protein_sequences, 
            add_special_tokens=True, 
            padding=True, 
            is_split_into_words=True, 
            return_tensors="pt")
        return outputs['input_ids'] 
    return tokenizer_function

def build_ankh_model_wrapper(_model):
    return AnkhWrapper(_model)


### progen

def create_progen_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def create_progen_model(model_path, type = "classification"):
    if type == "causal" :
        return ProGenForCausalLM.from_pretrained(model_path)
    elif type == "classification" :
        return ProGenForSequenceClassification.from_pretrained(model_path)

def build_avg_pooling_model(_base_model):
    # pdb.set_trace()
    if 'esm' in _base_model:
            cfg = load_data(_base_model)["cfg"]["model"]
            base_model = ESM2(
                num_layers=cfg.encoder_layers,
                embed_dim=cfg.encoder_embed_dim,
                attention_heads=cfg.encoder_attention_heads,
                token_dropout=cfg.token_dropout,
            )
            # model_config = AutoConfig.from_pretrained(f"facebook/{_base_model}")
            tokenizer = build_base_esm_tokenizer(_base_model)
            model = AvgPoolingModel(base_model)
    elif 'prot_t5' in _base_model:
        tokenizer = T5Tokenizer.from_pretrained(_base_model, legacy=False)
        print("tokenizer loaded")
        base_model = T5EncoderModel.from_pretrained(_base_model)
        # model_config = AutoConfig.from_pretrained(_base_model)
        model = AvgPoolingModel(base_model)
    elif 'progen' in _base_model:
        tokenizer = create_progen_tokenizer_custom(file='/home/rsawhney/progen/progen2/tokenizer.json')
        tokenizer.enable_truncation(max_length=1024)
        print("tokenizer loaded")
        # model_config = { "model_type": "progen"}
        base_model = create_progen_model(_base_model, type='causal')
        model = AvgPoolingModel(base_model)
        print(model)
        print("model_loaded")
    elif 'ankh' in _base_model:
        match = re.search(r"ankh_(base_model|large_model)", _base_model)
        if match:
            result = match.group(1)
            attr_name = 'load_' + result
            load_function = getattr(ankh, attr_name)
            model, tokenizer = load_function()
            tokenizer = build_base_ankh_tokenizer(tokenizer)
            base_model = build_ankh_model_wrapper(model)
            model = AvgPoolingModel(base_model)
        else:
            raise
    else:
        raise 

    test_model(model, tokenizer)
    return model, tokenizer

def build_test_model(_base_model, embedding=False):
    cfg = load_data(_base_model)["cfg"]["model"]
    base_model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        token_dropout=cfg.token_dropout,
    )
    model = TestModel2(base_model, embedding)
    tokenizer = build_base_esm_tokenizer(_base_model)
    if not embedding:
        tokenizer = lambda x: x[0] #for training we will get raw embeddings
    return model, tokenizer

def test_model(model, tokenizer):
    # pdb.set_trace()
    seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYL RSLGYNIVATPRGYVLAGG"
    model_type = model.__dict__["model_type"]
    print('model_type = ', model_type)
    if model_type == 't5':
        seq = " ".join(seq)
        tokens = tokenizer([seq])
        input_ids = torch.tensor(tokens["input_ids"])
        x = model(input_ids)
    elif model_type == 'esm': 
        tokens = tokenizer([seq])
        x = model(tokens)
    elif model_type == 'progen':
        tokens = tokenizer.encode(seq)
        input_ids = torch.tensor(tokens.ids)
        x = model(input_ids)


class AnkhWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self,x):
        output = self.base_model(input_ids=x)['last_hidden_state']
        return output

class TestModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # # self.base_model = base_model
        # for param in self.base_model.parameters(): #freeze params
        #     param.requires_grad = False
        resnet = models.resnet18(pretrained=False)
        # Modify the first convolutional layer to accept single-channel input
        num_input_channels = 1
        num_output_channels = resnet.conv1.out_channels
        num_features = resnet.fc.in_features
        resnet.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size=7, stride=2, padding=3, bias=False)
        resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        custom_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            torch.nn.Flatten(),                 # Flatten the 2D feature map
            torch.nn.Linear(num_features, 1000) # Linear layer projecting to 1000-dimensional output
        )
        self.next_model = torch.nn.Sequential(
            resnet,
            custom_head
        )

    def forward(self,x):
        # with torch.no_grad():
        #     x = self.base_model(x)
        x = x.unsqueeze(1)
        x = self.next_model(x)
        return x

class TestModel3(torch.nn.Module):
    def __init__(self, base_model, input_length=1200):
        super().__init__()
        # # self.base_model = base_model
        # for param in self.base_model.parameters(): #freeze params
        #     param.requires_grad = False

        x_dim = base_model.embed_dim
        projection_dim = x_dim//4
        self.projection_layer = torch.nn.Linear(x_dim, projection_dim)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(projection_dim, projection_dim)
        self.relu2 = torch.nn.ReLU()
        self.input_length = input_length
        num_features = input_length*projection_dim
        self.output = torch.nn.Linear(num_features, 1000) # Linear layer projecting to 1000-dimensional output

    def forward(self,x):
        # with torch.no_grad():
        #     x = self.base_model(x)
        num_rows_to_pad = self.input_length - x.shape[1]
        x = torch.nn.functional.pad(x, (0, 0, 0, num_rows_to_pad))
        
        x = self.projection_layer(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = x.view(x.shape[0], -1)
        x = self.output(x)
        return x

class TestModel2(torch.nn.Module):
    def __init__(self, base_model, embedding, input_length=1200):
        super().__init__()
        self.embedding = embedding
        if self.embedding:
            self.base_model = base_model
            for param in self.base_model.parameters(): #freeze params
                param.requires_grad = False

        x_dim = base_model.embed_dim
        projection_dim = x_dim//64
        self.projection_layer = torch.nn.Linear(x_dim, projection_dim)
        # self.relu1 = torch.nn.ReLU()
        # self.input_length = input_length
        l = torch.nn.TransformerEncoderLayer(projection_dim, 2, dim_feedforward=200, dropout=0.7)
        self.t = torch.nn.TransformerEncoder(l, 2)
        self.uses_embedding = False

    def forward(self,x):
        if self.embedding:
            with torch.no_grad():
                x = self.base_model(x)
        # num_rows_to_pad = self.input_length - x.shape[1]
        # x = torch.nn.functional.pad(x, (0, 0, 0, num_rows_to_pad))
        x = self.projection_layer(x)
        # x = self.relu1(x)
        x = self.t(x)
        return x[0][0].unsqueeze(0)

class AvgPoolingModel(torch.nn.Module):
    def __init__(self, base_model):
        # pdb.set_trace()
        super().__init__()
        self.base_model = base_model
        self.uses_embedding=False
        self.model_type = self.base_model.config.__dict__["model_type"] if type(base_model) is not ESM2 else "esm"
        if self.model_type == 't5':
            self.layer_devices = self.init_layer_devices()

    def forward(self,x):
        if self.model_type == 't5':
            mod_out = self.base_model(x) # output_hidden_states=True for all layers
            x = mod_out["last_hidden_state"]
        elif self.model_type == 'esm':
            mod_out = self.base_model(x)
            x = mod_out # already last layer
        elif self.model_type == 'progen':
            mod_out = self.base_model(x, output_hidden_states=True)
            x = mod_out["hidden_states"][-1].unsqueeze(0)
        x = torch.mean(x, dim=1)
        return x
    
    def init_layer_devices(self) -> Optional[List[torch.device]]:
        try:
            return [list(layer.parameters())[0].device for layer in self.base_model.encoder.block]
        except AttributeError:
            return None

    def update_layer_devices(self):
        # pdb.set_trace()
        self.layer_devices = self.init_layer_devices()
        for idx, layer in enumerate(self.base_model.encoder.block):
            if self.layer_devices and idx < len(self.layer_devices):
                layer.to(self.layer_devices[idx])

        
# def init_layer_devices(model: nn.Module) -> Optional[List[torch.device]]:
#     try:
#         return [list(layer.parameters())[0].device for layer in model.encoder.block]
#     except AttributeError:
#         return None

# def update_layer_devices(model: nn.Module):
#     model.layer_devices = init_layer_devices(model)
#     for idx, layer in enumerate(model.encoder.block):
#         if model.layer_devices and idx < len(model.layer_devices):
#             layer.to(model.layer_devices[idx])

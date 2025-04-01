import torch
from torch import nn
import loralib as lora
from collections import deque

SUPPORTED_OPERATORS = [nn.Linear]
LORA_RANK = 3

def get_layer_init_params(layer):
    if isinstance(layer, nn.Linear):
        _in = layer.in_features
        _out = layer.out_features
        return _in, _out
    elif isinstance(layer, nn.Embedding):
        _in = layer.in_features
        _out = layer.out_features
        return _in, _out
    elif isinstance(layer, nn.Conv2d):
        _in = layer.in_channels
        _out = layer.out_channels
        return _in, _out
    else:
        raise

def collect_modules_to_replace(module, target_layer_types):
    modules_to_replace = []
    for name, child_module in module.named_children():
        if len(list(child_module.named_children())) == 0:  # Check if the module has no children
            modules_to_replace.append((module, name, child_module))
        modules_to_replace.extend(collect_modules_to_replace(child_module, target_layer_types))

    return modules_to_replace

def process_layer(layer, parent, name, custom_module, use_current_weights, use_lora_zeros):
    if not hasattr(layer, 'weight'):
        use_current_weights = False
    if use_current_weights:
        layer_weight, layer_bias = layer.weight, layer.bias
    new_layer_type = getattr(custom_module, layer.__class__.__name__)
    _in, _out = get_layer_init_params(layer)  # Implement get_layer_init_params function
    new_layer = new_layer_type(_in, _out, r=LORA_RANK)  # Replace with actual parameters
    if use_current_weights:
        new_layer._parameters['weight'] = layer_weight
        new_layer._parameters['bias'] = layer_bias
    if use_lora_zeros:
        shape_a = new_layer._parameters['lora_A'].shape
        shape_b = new_layer._parameters['lora_B'].shape
        new_layer._parameters['lora_A'] = torch.zeros(shape_a)
        new_layer._parameters['lora_B'] = torch.zeros(shape_b)
    setattr(parent, name, new_layer)
    return parent

def construct_parent_child_map(model):
    parent_child_map = {}
    queue = deque([model])
    while queue:
        module = queue.popleft()
        for child_name, child_module in module.named_children():
            parent_child_map[child_module] = (module, child_name)
            queue.append(child_module)

    return parent_child_map

def get_starting_children(model, target_layer_types):
    ret = []
    queue = deque([model])
    while queue:
        module = queue.popleft()
        if not list(module.named_children()):
            if any(isinstance(module, layer_type) for layer_type in target_layer_types):
                ret.append(module)
        else:
            for child_name, child_module in module.named_children():
                queue.append(child_module)
    return ret

def replace_layers_with_custom_module(model, target_layer_types, custom_module, use_current_weights=True, use_lora_zeros=False):
    # Collect parent-child relationships
    parent_child_map = construct_parent_child_map(model)
    
    # Create a queue and populate it with leaf modules to start the process
    queue = deque()
    starters = get_starting_children(model, target_layer_types)
    for s in starters:
        queue.append(s)
    # Process the queue
    replaced = {}
    while queue:
        module = queue.popleft()
        tup = parent_child_map.get(module)
        if tup:
            parent, child_name = tup[0], tup[1]
            if any(isinstance(module, layer_type) for layer_type in target_layer_types):
                modified_parent = process_layer(module, parent, child_name, custom_module, use_current_weights, use_lora_zeros)
            else:
                setattr(parent, child_name, module)
            
            queue.append(parent)
    
def convert_model(model):
    replace_layers_with_custom_module(model, SUPPORTED_OPERATORS, lora, use_lora_zeros=False)
    return model

# from model_zoo import build_avg_pooling_model

# model, tokenizer = build_avg_pooling_model()
# input = ['AAAAAAAA']
# input = tokenizer(input)
# output = model(input)
# print(f'before {output}')
# import time 
# model.cuda()
# time.sleep(10)
# model = convert_model(model)
# model.cuda()
# time.sleep(10)
# model.train(False)
# print(f'after  {model(input)}')
# import inspect
# print(model.base_model.contact_head.regression._parameters['lora_B'].shape, model.base_model.contact_head.regression.scaling)
# print(inspect.getsource(model.base_model.contact_head.regression.forward))
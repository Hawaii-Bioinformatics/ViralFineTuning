import pdb
from typing import List, Optional

import torch

from collections import defaultdict, deque

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.param_count = None
        self.layer = None
        self.device_id = None

def construct_tree(node_name, module_graph, name_layer_map):
    node = TreeNode(node_name)
    layer = name_layer_map[node_name]
    node.layer = layer
    if node_name in module_graph:
        for child_name in module_graph[node_name]:
            child_node = construct_tree(child_name, module_graph, name_layer_map)
            node.children.append(child_node)
    return node

def get_param_count(layer):
    param_count = sum(p.numel() for p in layer.parameters())
    return param_count

def calculate_param_counts(node):
    if not node.children:
        node.param_count = get_param_count(node.layer)
    else:
        for child in node.children:
            calculate_param_counts(child)
        node.param_count = sum(child.param_count for child in node.children)

def print_tree(node, depth=0):
    if node:
        if node.device_id is not None:
            print('  ' * depth + f"{node.name} (param_count: {node.param_count}, device: {node.device_id})")
        for child in node.children:
            print_tree(child, depth + 1)

def construct_parent_child_map(model):
    parent_child_map = defaultdict(list)
    name_layer_map = {}
    queue = deque([('', model)])
    while queue:
        module = queue.popleft()
        module_name, module_obj = module[0], module[1]
        name_layer_map[module_name] = module_obj
        for child_name, child_module in module_obj.named_children():
            parent_child_map[module_name].append(child_name)
            queue.append((child_name, child_module))
    return parent_child_map, name_layer_map

def partition_layers(node, n, e=0.005):
    partition_amt = [0 for _ in range(n)]
    partition_limit = (node.param_count/n)*(1+e)
    print(f'using limit: {partition_limit}')
    children = node.children
    i = 0
    while i < len(children):
        node = children[i]
        if node.param_count >= partition_limit:
            children = children[:i] + node.children + children[i+1:] # Replace curr node w/ children
        else:
            i += 1

    current_partition = 0
    while children:
        child = children[0]
        if child.param_count + partition_amt[current_partition] > partition_limit:
            # last partition is full
            current_partition += 1
        else:
            partition_amt[current_partition] += child.param_count
            child.device_id = current_partition
            children = children[1:]
    print(f'partitioning parameters as follows:')
    for i in range(len(partition_amt)):
        print(f'device {i}: {partition_amt[i]} parameters')

def send_sub_model_to_device(node, device):
    #avoid overwriting if weights are shared
    for param in node.layer.parameters(): 
        if param.device.type == 'cuda':
            device = param.device
    
    node.layer.to(device)
    for child in node.children:
        send_sub_model_to_device(child, device)

def send_model_to_devices(node, devices):
    if node.device_id is not None:
        send_sub_model_to_device(node, devices[node.device_id])
    else:
        for child_module in node.children:
            send_model_to_devices(child_module, devices)

def split_model_over_multiple_devices(model, devices):
    # pdb.set_trace()
    root_module = 'base_model'
    model_type = model.__dict__["model_type"]
    if model_type == 'esm':
        root_module = 'base_model'
    elif model_type == 't5':
        root_module = 'base_model'
    module_graph, name_layer_map = construct_parent_child_map(model)
    tree = construct_tree(root_module, module_graph, name_layer_map, set())
    calculate_param_counts(tree)
    partition_layers(tree, len(devices))
    send_model_to_devices(tree, devices)
    if model_type == 'esm':
        model.base_model._init_layer_devices()
    elif model_type == 't5':
        model.update_layer_devices()
    print('param distribution:')
    print_tree(tree)


# def init_layer_devices(model: nn.Module) -> Optional[List[torch.device]]:
#     try:
#         # Retrieve the device of the first parameter of each layer
#         return [list(layer.parameters())[0].device for layer in model.layers]
#     except AttributeError:
#         # If 'layers' attribute does not exist or an error occurs
#         return None

# import torch
# from torch import nn
# from collections import defaultdict, deque

# class TreeNode:
#     def __init__(self, name):
#         self.name = name
#         self.children = []
#         self.param_count = None
#         self.layer = None
#         self.device_id = None
# # new
# def construct_tree(root_name, module_graph, name_layer_map):
#     # Create the root node
#     root = TreeNode(root_name)
#     root.layer = name_layer_map[root_name]

#     # Use a stack to manage nodes to process
#     stack = [(root, root_name)]
#     visited = set()  # To track visited nodes

#     while stack:
#         parent_node, parent_name = stack.pop()

#         if parent_name in visited:
#             continue

#         visited.add(parent_name)

#         # If the parent has children, add them to the stack
#         if parent_name in module_graph:
#             for child_name in module_graph[parent_name]:
#                 if child_name not in visited:
#                     child_node = TreeNode(child_name)
#                     child_node.layer = name_layer_map[child_name]
#                     parent_node.children.append(child_node)
#                     stack.append((child_node, child_name))

#     return root

# # def construct_tree(node_name, module_graph, name_layer_map):
# #     # OG
# #     # pdb.set_trace()
# #     node = TreeNode(node_name)
# #     layer = name_layer_map[node_name]
# #     node.layer = layer
# #     if node_name in module_graph:
# #         for child_name in module_graph[node_name]:
# #             child_node = construct_tree(child_name, module_graph, name_layer_map)
# #             node.children.append(child_node)
# #     return node

# def get_param_count(layer):
#     param_count = sum(p.numel() for p in layer.parameters())
#     return param_count

# def calculate_param_counts(node):
#     if not node.children:
#         node.param_count = get_param_count(node.layer)
#     else:
#         for child in node.children:
#             calculate_param_counts(child)
#         node.param_count = sum(child.param_count for child in node.children)

# def print_tree(node, depth=0):
#     if node:
#         if node.device_id is not None:
#             print('  ' * depth + f"{node.name} (param_count: {node.param_count}, device: {node.device_id})")
#         for child in node.children:
#             print_tree(child, depth + 1)

# def construct_parent_child_map(model):
#     parent_child_map = defaultdict(list)
#     name_layer_map = {}
#     queue = deque([('', model)])
#     while queue:
#         module_name, module_obj = queue.popleft()
#         name_layer_map[module_name] = module_obj
#         for child_name, child_module in module_obj.named_children():
#             parent_child_map[module_name].append(child_name)
#             queue.append((child_name, child_module))
#     return parent_child_map, name_layer_map

# # OG
# # def partition_layers(node, n, e=0.005):
# #     partition_amt = [0 for _ in range(n)]
# #     partition_limit = (node.param_count/n)*(1+e)
# #     print(f'using limit: {partition_limit}')
# #     children = node.children
# #     i = 0
# #     while i < len(children):
# #         node = children[i]
# #         if node.param_count >= partition_limit:
# #             children = children[:i] + node.children + children[i+1:] # Replace curr node w/ children
# #         else:
# #             i += 1

# #     current_partition = 0
# #     while children:
# #         child = children[0]
# #         if child.param_count + partition_amt[current_partition] > partition_limit:
# #             # last partition is full
# #             current_partition += 1
# #         else:
# #             partition_amt[current_partition] += child.param_count
# #             child.device_id = current_partition
# #             children = children[1:]
# #     print(f'partitioning parameters as follows:')
# #     for i in range(len(partition_amt)):
# #         print(f'device {i}: {partition_amt[i]} parameters')

# # NEW - updated
# def partition_layers(node, num_devices):
#     total_params = node.param_count
#     device_limits = [total_params // num_devices] * num_devices
#     for i in range(total_params % num_devices):
#         device_limits[i] += 1

#     device_id = 0
#     remaining_capacity = device_limits[device_id]

#     def assign_device(node, device_id, remaining_capacity):
#         if not node.children:  # Leaf node
#             node.device_id = device_id
#             remaining_capacity -= node.param_count
#             return remaining_capacity

#         for child in node.children:
#             if remaining_capacity < child.param_count and device_id + 1 < num_devices:
#                 device_id += 1
#                 remaining_capacity = device_limits[device_id]
#             remaining_capacity = assign_device(child, device_id, remaining_capacity)
#         node.device_id = device_id  # Assign current node to the same device as its children
#         return remaining_capacity

#     assign_device(node, device_id, remaining_capacity)

# def send_sub_model_to_device(node, device):
#     #avoid overwriting if weights are shared
#     for param in node.layer.parameters():
#         if param.device.type == 'cuda':
#             device = param.device
    
#     node.layer.to(device)
#     for child in node.children:
#         send_sub_model_to_device(child, device)

# def send_model_to_devices(node, devices):
#     if node.device_id is not None:
#         send_sub_model_to_device(node, devices[node.device_id])
#     else:
#         for child_module in node.children:
#             send_model_to_devices(child_module, devices)

# def split_model_over_multiple_devices(model, model_config, devices):
#     pdb.set_trace()
#     module_graph, name_layer_map = construct_parent_child_map(model)
#     root_module = 'base_model'

#     tree = construct_tree(root_module, module_graph, name_layer_map)
#     calculate_param_counts(tree)
#     partition_layers(tree, len(devices))
#     send_model_to_devices(tree, devices)
#     if model_config.model_type == 'esm':
#         model.base_model._init_layer_devices()
#     # elif model_config.model_type == 't5':
#         # init_layer_devices(model)
#     print('param distribution:')
#     print_tree(tree)

# def init_layer_devices(model):
#     """
#     Assign each layer of the T5 model to a specific device.
    
#     Args:c
#         t5_model (T5Model): An instance of the T5 model.
#     """
#     pdb.set_trace()
#     layer_devices = []
    
#     # Assign encoder layers to devices
#     for i, layer in enumerate(model.encoder.block):
#         device = torch.device(f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
#         layer.to(device)
#         layer_devices.append(device)
    
#     # # Assign decoder layers to devices
#     # for i, layer in enumerate(t5_model.decoder.block):
#     #     device = torch.device(f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
#     #     layer.to(device)
#     #     layer_devices.append(device)
    
#     return layer_devices
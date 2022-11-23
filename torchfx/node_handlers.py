import torch
import torch.fx as fx
from torch.fx.node import Node, _get_qualified_name

from utils import get_leaf_node

def linear_handler(gm: fx.GraphModule, node: Node):
    args = node.meta["args"]
    leaf = get_leaf_node(gm, node)
    in_channel = leaf.in_features
    out_channel = leaf.out_features

    print(in_channel, out_channel)

    return 0

def conv2d_handler(gm: fx.GraphModule, node: Node):
    return 0

def batchnorm2d_handler(gm: fx.GraphModule, node: Node):
    return 0

def relu_handler(gm: fx.GraphModule, node: Node):
    return 0

def flatten_handler(gm: fx.GraphModule, node: Node):
    return 0

def adaptiveavgpool2d_handler(gm: fx.GraphModule, node: Node):
    return 0

def operator_add_handler(gm: fx.GraphModule, node: Node):
    return 0
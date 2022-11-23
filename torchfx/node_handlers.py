"""
op handlers
returns None if flops cannot be calculated by the given node.

the formulae are taken from 
https://github.com/sovrasov/flops-counter.pytorch/blob/master/ptflops/pytorch_ops.py
"""
import numpy as np
import torch
import torch.fx as fx
from torch.fx.node import Node, _get_qualified_name

from utils import get_leaf_node

def check_tensor_args(node: Node, tensor_cnt):
    args = node.meta["args"]
    if args is None or not (type(args) == tuple or type(args) == list):
        return False
    if len(args) < tensor_cnt:
        return False
    for i in range(tensor_cnt):
        if not torch.is_tensor(args[i]):
            return False
    return True

def linear_handler(gm: fx.GraphModule, node: Node):
    if not check_tensor_args(node, 1):
        return

    args = node.meta["args"]
    leaf = get_leaf_node(gm, node)
    in_tensor: torch.Tensor = args[0]

    out_channel = leaf.out_features
    flops = in_tensor.nelement() * out_channel

    # TODO: add bias (but the bias should be merged to gemm, so I don't think it should be added)

    return flops

def conv2d_handler(gm: fx.GraphModule, node: Node):
    if not check_tensor_args(node, 1):
        return

    args = node.meta["args"]
    leaf = get_leaf_node(gm, node)
    in_tensor: torch.Tensor = args[0]

    # TODO: what if the alignment of in_tensor is not an NCHW format?
    (batch_size, in_channels, h, w) = tuple(in_tensor.shape)
    out_channels = leaf.out_channels
    groups = leaf.groups
    kernel_dims = list(leaf.kernel_size)
    
    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * h * w
    overall_conv_flops = conv_per_position_flops * active_elements_count

    # should I add bias?
    # if conv_module.bias is not None:
    #     bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops)


def batchnorm2d_handler(gm: fx.GraphModule, node: Node):
    if not check_tensor_args(node, 1):
        return

    args = node.meta["args"]
    leaf = get_leaf_node(gm, node)
    in_tensor: torch.Tensor = args[0]

    flops = in_tensor.nelement()
    if leaf.affine:
        flops *= 2
    return flops

def relu_handler(gm: fx.GraphModule, node: Node):
    if not check_tensor_args(node, 1):
        return
    args = node.meta["args"]

    return args[0].nelement()

def flatten_handler(gm: fx.GraphModule, node: Node):
    return 0

# TODO: does every pool have same flops?
def pool_handler(gm: fx.GraphModule, node: Node):
    if not check_tensor_args(node, 1):
        return

    args = node.meta["args"]
    in_tensor: torch.Tensor = args[0]

    return in_tensor.nelement()

def operator_add_handler(gm: fx.GraphModule, node: Node):
    if not check_tensor_args(node, 2):
        return

    args = node.meta["args"]
    ret_shape = torch.broadcast_shapes(args[0].shape, args[1].shape)

    return ret_shape.numel()

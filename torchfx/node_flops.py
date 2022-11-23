from typing import Any, Dict

import torch
import torch.fx as fx
from torch.fx.node import Node, _get_qualified_name

from node_handlers import *
from utils import node_name, args_str

# Some modules use more faster kernel APIs. (e.g. linear & tensor core)
# the weights will be multiplied to FLOPs to get FLOPS (FLOPs per sec)
# TODO: find handler by op name
MODULE_FLOPS_HANDLER_WEIGHTS = {
    'torch.nn.modules.conv.Conv2d': (conv2d_handler, 1.0),
    'torch.nn.modules.batchnorm.BatchNorm2d': (batchnorm2d_handler, 1.0),
    'torch.nn.modules.pooling.AdaptiveAvgPool2d': (adaptiveavgpool2d_handler, 1.0),
    'torch.nn.modules.linear.Linear': (linear_handler, 1.0),
    'torch.nn.modules.activation.ReLU': (relu_handler, 1.0),
    'torch.flatten': (flatten_handler, 1.0),
    '_opeartor.add': (operator_add_handler, 1.0),
    # '_opeartor.mul': 1.0
}

def node_flops(gm: fx.GraphModule, node: Node):
    if "args" not in node.meta:
        return

    if node.op == 'call_module' or node.op == 'call_function':
        fn_name = node_name(gm, node)
        if fn_name not in MODULE_FLOPS_HANDLER_WEIGHTS:
            return
        handler, weight = MODULE_FLOPS_HANDLER_WEIGHTS[fn_name]
        flops = handler(gm, node)
        node.meta["flops"] = flops * weight
    

class FxFlopsAdder(fx.interpreter.Interpreter):
    def run_node(self, n: Node) -> Any:
        # with fx_traceback.append_stack_trace(n.stack_trace):
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        print(f"run_node {n.op}, {n.target} got args {args_str(args)}")
        n.meta["args"] = args
        n.meta["kwargs"] = kwargs

        print_meta: Dict[str, str] = dict()

        print_meta['input'] = args_str(args)
        print_meta['flops'] = node_flops(self.module, n)

        n.meta['print_meta'] = print_meta
        
        return getattr(self, n.op)(n.target, args, kwargs)
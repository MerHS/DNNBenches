from typing import Any

import torch
import torch._dynamo as dynamo
import torch._dynamo.logging
from functorch.compile import aot_function
from graph_drawer import FxGraphDrawer

import torch.fx as fx
from torch.fx.node import Node, _get_qualified_name

def args_str(args, newline=True):
    sep = ',\n' if newline else ', '
    # a debug helper
    if torch.is_tensor(args):
        return f"{args.dtype}{str(list(args.shape))}"
    elif isinstance(args, tuple):
        joined = sep.join([args_str(x) for x in args])
        return f"tuple({joined})"
    elif isinstance(args, list):
        joined = sep.join([args_str(x) for x in args])
        return f"list({joined})"
    else:
        return str(args)


def get_leaf_node(module: torch.nn.Module, node: Node) -> torch.nn.Module:
    py_obj = module
    assert isinstance(node.target, str)
    atoms = node.target.split(".")
    for atom in atoms:
        if not hasattr(py_obj, atom):
            raise RuntimeError(f"{str(py_obj)} does not have attribute {atom}!")
        py_obj = getattr(py_obj, atom)
    return py_obj

def typename(target: Any) -> str:
    if isinstance(target, torch.nn.Module):
        return torch.typename(target)
    elif isinstance(target, str):
        return target
    else:
        return _get_qualified_name(target)

def node_name(gm: fx.GraphModule, node: Node):
    if node.op == 'call_module':
        return typename(get_leaf_node(gm, node))
    elif node.op == 'call_function':
        return typename(node.target)
    else:
        return ""

def print_graph(model, model_name, args, kwargs):
    def printer(gm: torch.fx.GraphModule, example_inputs):
        print("print fx graph")
        draw = FxGraphDrawer(gm, "gpt2")
        dot = draw.get_dot_graph()
        with open(f"{model_name}.svg", "wb") as f:
            f.write(dot.create_svg())
        return gm.forward  # return a python callable

    @dynamo.optimize(printer)
    def loop(args, kwargs):
        model(*args, kwargs)

    loop(args, kwargs)

def print_aot(model, model_name, args, kwargs):
    model.train()

    def print_forward(gm: torch.fx.GraphModule, example_inputs):
        print("print forward aot graph")
        draw = FxGraphDrawer(gm, "gpt2")
        dot = draw.get_dot_graph()
        with open(f"{model_name}_aot_forward.svg", "wb") as f:
            f.write(dot.create_svg())
        return gm

    def print_backward(gm: torch.fx.GraphModule, example_inputs):
        print("print backward aot graph")
        draw = FxGraphDrawer(gm, "gpt2")
        dot = draw.get_dot_graph()
        with open(f"{model_name}_aot_backward.svg", "wb") as f:
            f.write(dot.create_svg())
        return gm

    # Pass on the compiler_fn to the aot_function API
    def loop(args, kwargs):
        return model(*args, **kwargs)

    aot_fn = aot_function(loop, fw_compiler=print_forward, bw_compiler=print_backward)

    kwargs = {
        k : v.clone().detach() for k, v in kwargs.items()
    }

    args = (v.clone().detach() for v in args)
    
    result = aot_fn(args, kwargs)
    
    return result

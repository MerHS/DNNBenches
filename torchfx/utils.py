import torch
import torch._dynamo as dynamo
import torch._dynamo.logging
from functorch.compile import aot_function
from graph_drawer import FxGraphDrawer

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

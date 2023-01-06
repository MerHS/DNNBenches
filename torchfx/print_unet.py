from torch.fx import symbolic_trace
from graph_drawer import FxGraphDrawer
from fx_utils import get_model
from pathlib import Path

import functorch.compile
from functorch.compile import aot_function
from functorch.compile import aot_module_simplified

# functorch.compile.config.use_functionalize = True
# functorch.compile.config.use_fake_tensor = True

cur_dir = Path(__file__)
cur_dir = cur_dir.parent.parent.absolute()

import torch._dynamo as dynamo

model, test_input = get_model("pytorch_unet", "cuda:0", 2)

def print_eager(gm, example_inputs):
    print("print eager graph")
    draw = FxGraphDrawer(gm, "unet")
    dot = draw.get_dot_graph()
    svg = dot.create_svg()
    with (cur_dir / "unet.svg").open("wb") as f:
        f.write(svg)
    return gm


def print_forward(gm, example_inputs):
    print("print forward aot graph")
    draw = FxGraphDrawer(gm, "unet")
    dot = draw.get_dot_graph()
    svg = dot.create_svg()
    with  (cur_dir / "unet_forward_only.svg").open("wb") as f:
        f.write(svg)
    print("export forward")
    return gm

def print_backward(gm, example_inputs):
    print("print backward aot graph")
    draw = FxGraphDrawer(gm, "unet")
    dot = draw.get_dot_graph()
    with (cur_dir / "unet_no_backward.svg").open("wb") as f:
        f.write(dot.create_svg())
    print("export backward")
    return gm    

# mod = dynamo.optimize(print_eager)(model)
# mod(*test_input)

def loop(batch):
    output = model(batch)
    loss = output.sum()
    return loss

# test = test_input[0].to(device="meta").requires_grad_(True)

def print_aot(gm, example_inputs):
    print("print unet graph")
    draw = FxGraphDrawer(gm, "unet")
    dot = draw.get_dot_graph()
    svg = dot.create_svg()
    with (cur_dir / "unet.svg").open("wb") as f:
        f.write(svg)

    cg = aot_module_simplified(gm, test_input, fw_compiler=print_forward, bw_compiler=print_backward)
    return cg

# aot_fn = aot_module_simplified(model, test_input, fw_compiler=print_forward, bw_compiler=print_backward)
# loss = aot_fn(test_input[0])
# loss.backward()

# model.train()
model.eval()
mod = dynamo.optimize(print_aot)(model)
loss = mod(*test_input)
# loss.sum().backward()
import torch
from torchvision.models import resnet50
from torch.fx import symbolic_trace
from graph_drawer import FxGraphDrawer

from node_flops import FxFlopsAdder

model = resnet50()
traced = symbolic_trace(model)

test_input = torch.rand((16, 3, 244, 244))

flops = FxFlopsAdder(traced)
flops.run(test_input)

g = FxGraphDrawer(traced, "resnet50")
dot = g.get_dot_graph()



with open("resnet.svg", "wb") as f:
    f.write(dot.create_svg())
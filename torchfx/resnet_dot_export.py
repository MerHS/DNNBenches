import torch
from torchvision.models import resnet50
from torch.fx import symbolic_trace
from graph_drawer import FxGraphDrawer

model = resnet50()
traced = symbolic_trace(model)

g = FxGraphDrawer(traced, "resnet50")
dot = g.get_dot_graph()
print(dot)
with open("resnet.svg", "wb") as f:
    f.write(dot.create_svg())
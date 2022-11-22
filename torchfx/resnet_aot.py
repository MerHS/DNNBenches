import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
import torch.optim as optim

from functorch.compile import aot_function
from graph_drawer import FxGraphDrawer
from utils import print_aot

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)


dataiter = iter(trainloader)
images, labels = next(dataiter)

model = resnet50()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
model_name = 'resnet50-grad'

def print_forward(gm: torch.fx.GraphModule, example_inputs):
    print("print forward aot graph")
    draw = FxGraphDrawer(gm, model_name)
    # dot = draw.get_dot_graph()
    # print("get dot")
    # svg = dot.create_svg()
    # print("get svg")
    # with open(f"{model_name}_aot_forward.svg", "wb") as f:
    #     f.write(svg)
    print("export forward")
    return gm

def print_backward(gm: torch.fx.GraphModule, example_inputs):
    print("print backward aot graph")
    draw = FxGraphDrawer(gm, model_name)
    dot = draw.get_dot_graph()
    with open(f"{model_name}_aot_backward.svg", "wb") as f:
        f.write(dot.create_svg())
    print("export backward")
    return gm

# Pass on the compiler_fn to the aot_function API
def loop(batch, label):
    output = model(batch)
    loss = criterion(output, label)
    return loss

aot_fn = aot_function(loop, fw_compiler=print_forward, bw_compiler=print_backward)
loss = aot_fn(images.requires_grad_(True), labels)
loss.backward()
print(loss)

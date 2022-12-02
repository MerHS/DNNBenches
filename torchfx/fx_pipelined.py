import torch
import torch.nn as nn
import torch.optim as optim
import torch.fx as fx
from utils import print_aot
from functorch.compile import aot_function
from torch.distributed.pipeline.sync import Pipe

import os

# Need to initialize RPC framework first.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

def main1():
    class PipeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = nn.Linear(50, 100).to('cuda:0')
            self.mod2 = nn.Linear(100, 50).to('cuda:1')
        
        def forward(self, x):
            x = self.mod1(x)
            x = x.to('cuda:1')
            x = self.mod2(x)
            return x

    model = PipeModule()
    opt = optim.Adam(model.parameters(), lr=1)

    # print(list(model.mod1.parameters()))

    opt.zero_grad()
    test_input = torch.rand(20, 50).to('cuda:0')
    test_target = torch.rand(20, 50).to('cuda:1')
    test_output = model(test_input)

    crit = nn.L1Loss()
    loss = crit(test_target, test_output)
    loss.backward()
    opt.step()

    print(loss.shape, loss.device)
    print(list(model.mod1.parameters()))

    traced = fx.symbolic_trace(model)
    traced.print_readable()
    print(str(traced.graph))

def main2():
    class ShareModule(nn.Module):
        def __init__(self, bias, idx):
            super().__init__()
            self.device = f'cuda:{idx}'
            self.bias = bias
            self.mod = nn.Linear(50, 50).to(self.device)

        def forward(self, x):
            x = x.to(self.device)
            x = self.mod(x)
            bias = self.bias.to(self.device)
            return x + bias

    bias = torch.rand(50).requires_grad_(True)
    mod1 = ShareModule(bias, 0).to('cuda:0')
    mod2 = ShareModule(bias, 1).to('cuda:1')
    model = nn.Sequential(mod1, mod2)

    opt = optim.Adam(list(model.parameters()) + [bias], lr=1)
    print(bias)

    opt.zero_grad()
    test_input = torch.rand(20, 50).to('cuda:0')
    test_target = torch.rand(20, 50).to('cuda:1')
    test_output = model(test_input)

    crit = nn.L1Loss()
    loss = crit(test_target, test_output)
    loss.backward()
    opt.step()

    print(loss.shape, loss.device)
    print(bias)

    traced = fx.symbolic_trace(model)
    traced.print_readable()
    print(str(traced.graph))


def main3():
    class ShareModule(nn.Module):
        def __init__(self, bias, idx):
            super().__init__()
            self.device = f'cuda:{idx}'
            self.bias = bias
            self.mod = nn.Linear(50, 50).to(self.device)

        def forward(self, x):
            # x = x.to(self.device)
            x = self.mod(x)
            # cannot comment it: bias is not moved automatically
            bias = self.bias.to(self.device) 
            return x + bias

    bias = torch.rand(50).requires_grad_(True)
    mod1 = ShareModule(bias, 0).to('cuda:0')
    mod2 = ShareModule(bias, 1).to('cuda:1')
    seq = nn.Sequential(mod1, mod2)
    model = Pipe(seq, chunks=2)

    opt = optim.Adam(list(model.parameters()) + [bias], lr=1)
    print(bias)

    opt.zero_grad()
    test_input = torch.rand(20, 50).to('cuda:0')
    test_target = torch.rand(20, 50).to('cuda:1')
    test_output = model(test_input)

    crit = nn.L1Loss()
    loss = crit(test_target, test_output.local_value())
    loss.backward()
    opt.step()

    print(loss.shape, loss.device)
    print(bias)

    # CANNOT TRACE PIPE
    # traced = fx.symbolic_trace(model)
    # traced.print_readable()
    # print(str(traced.graph))

if __name__ == '__main__':
    main3()

# opt.zero_grad()

# test_output = traced(test_input)
# loss = crit(test_target, test_output)

# print(loss)

# loss.backward()
# opt.step()

# class OneTwo(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mod1 = nn.Linear(10, 50)
#         self.mod2 = nn.Linear(10, 50)
    
#     def forward(self, x):
#         return self.mod1(x), self.mod2(x)

# class TwoTwo(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mod1 = nn.Linear(50, 50)
#         self.mod2 = nn.Linear(50, 50)
    
#     def forward(self, x, y):
#         x = self.mod1(x)
#         y = self.mod2(y)
#         return x, y

# lin = nn.Sequential(OneTwo(), TwoTwo(), TwoTwo())
# x, y = lin(torch.rand(20, 10))
# print(x + y)

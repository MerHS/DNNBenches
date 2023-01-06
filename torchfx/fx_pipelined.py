import torch
import torch.nn as nn
import torch.optim as optim
import torch.fx as fx

from functorch.compile import aot_function, aot_module
import torch._dynamo as dynamo
from torch.distributed.pipeline.sync import Pipe

from graph_drawer import FxGraphDrawer
from fx_utils import print_aot
import os

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%y%m%d-%H%M%S")
result_path = f'{os.getcwd()}/result/pipe-test-{current_time}'

# Need to initialize RPC framework first.
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

def main_pipe():
    class PipeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = nn.Sequential(
                nn.Linear(50, 10000),
                *[nn.Linear(10000, 10000) for _ in range(10)]
            ).to('cuda:0')
            self.mod2 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(10)],
                nn.Linear(10000, 50)
            ).to('cuda:1')
        
        def forward(self, x):
            x = self.mod1(x)
            x = x.to('cuda:1')
            x = self.mod2(x)
            return x

    model = PipeModule()
    opt = optim.Adam(model.parameters(), lr=1)

    # print(list(model.mod1.parameters()))

    opt.zero_grad()
    test_input = torch.rand(4, 20, 50).to('cuda:0')
    test_target = torch.rand(4, 20, 50).to('cuda:1')
    crit = nn.L1Loss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(result_path, worker_name='worker0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for _ in range(10):
            opt.zero_grad()
            out_1 = model(test_input[:2, ...])
            loss_1 = crit(test_target[:2, ...], out_1)
            out_2 = model(test_input[2:, ...])
            loss_2 = crit(test_target[2:, ...], out_2)
            
            loss_2.backward()
            loss_1.backward()

            opt.step()

            p.step()


def main_pipe4():
    class PipeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = nn.Sequential(
                nn.Linear(50, 10000),
                *[nn.Linear(10000, 10000) for _ in range(10)]
            ).to('cuda:0')
            self.mod2 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(10)],
                nn.Linear(10000, 50)
            ).to('cuda:1')
        
        def forward(self, x):
            x = self.mod1(x)
            x = x.to('cuda:1')
            x = self.mod2(x)
            return x

    model = PipeModule()
    opt = optim.Adam(model.parameters(), lr=1)

    # print(list(model.mod1.parameters()))

    opt.zero_grad()
    test_input = torch.rand(4, 20, 50).to('cuda:0')
    test_target = torch.rand(4, 20, 50).to('cuda:1')
    crit = nn.L1Loss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(result_path, worker_name='worker0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for _ in range(10):
            opt.zero_grad()
            out_1 = model(test_input[:1, ...])
            out_2 = model(test_input[1:2, ...])
            out_3 = model(test_input[2:3, ...])
            out_4 = model(test_input[3:4, ...])
            loss_1 = crit(test_target[:1, ...], out_1)
            loss_2 = crit(test_target[1:2, ...], out_2)
            loss_3 = crit(test_target[2:3, ...], out_3)
            loss_4 = crit(test_target[3:4, ...], out_4)
            
            loss_4.backward()
            loss_3.backward()
            loss_2.backward()
            loss_1.backward()

            opt.step()

            p.step()



def main_pipe_bfs():
    class PipeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = nn.Sequential(
                nn.Linear(50, 10000),
                *[nn.Linear(10000, 10000) for _ in range(5)]
            ).to('cuda:0')
            self.mod2 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(5)]
            ).to('cuda:1')
            self.mod3 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(5)]
            ).to('cuda:0')
            self.mod4 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(5)],
                nn.Linear(10000, 50)
            ).to('cuda:1')
        
        def forward(self, x):
            x = self.mod1(x)
            x = x.to('cuda:1')
            x = self.mod2(x)
            x = x.to('cuda:0')
            x = self.mod3(x)
            x = x.to('cuda:1')
            x = self.mod4(x)
            return x

    model = PipeModule()
    opt = optim.Adam(model.parameters(), lr=1)

    # print(list(model.mod1.parameters()))

    opt.zero_grad()
    test_input = torch.rand(4, 20, 50).to('cuda:0')
    test_target = torch.rand(4, 20, 50).to('cuda:1')
    crit = nn.L1Loss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(result_path, worker_name='worker0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for _ in range(10):
            opt.zero_grad()
            out_1 = model(test_input[:2, ...])
            out_2 = model(test_input[2:, ...])
            loss_1 = crit(test_target[:2, ...], out_1)
            loss_2 = crit(test_target[2:, ...], out_2)
            
            loss_2.backward()
            loss_1.backward()

            opt.step()
            # torch.cuda.synchronize()

            p.step()


def main_hand_pipe_bfs():
    class PipeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = nn.Sequential(
                nn.Linear(50, 10000),
                *[nn.Linear(10000, 10000) for _ in range(5)]
            ).to('cuda:0')
            self.mod2 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(5)]
            ).to('cuda:1')
            self.mod3 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(5)]
            ).to('cuda:0')
            self.mod4 = nn.Sequential(
                *[nn.Linear(10000, 10000) for _ in range(5)],
                nn.Linear(10000, 50)
            ).to('cuda:1')
        
        def forward(self, x):
            x = self.mod1(x)
            x = x.to('cuda:1')
            x = self.mod2(x)
            x = x.to('cuda:0')
            x = self.mod3(x)
            x = x.to('cuda:1')
            x = self.mod4(x)
            return x

    model = PipeModule()
    opt = optim.Adam(model.parameters(), lr=1)

    # print(list(model.mod1.parameters()))

    opt.zero_grad()
    test_input = torch.rand(4, 20, 50).to('cuda:0')
    test_target = torch.rand(4, 20, 50).to('cuda:1')
    crit = nn.L1Loss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=5),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(result_path, worker_name='worker0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for _ in range(10):
            opt.zero_grad()

            # time step 1
            out_1_1 = model.mod1(test_input[:2, ...])

            # time step 2
            out_1_2 = model.mod2(out_1_1.to('cuda:1', non_blocking=True))
            out_2_1 = model.mod1(test_input[2:, ...])

            # time step 3
            out_1_3 = model.mod3(out_1_2.to('cuda:0', non_blocking=True))
            out_2_2 = model.mod2(out_2_1.to('cuda:1', non_blocking=True))

            # time step 4
            out_1_4 = model.mod4(out_1_3.to('cuda:1', non_blocking=True))
            loss_1 = crit(test_target[:2, ...], out_1_4)
            out_2_3 = model.mod3(out_2_2.to('cuda:0', non_blocking=True))

            # time step 5
            loss_1.backward()
            out_2_4 = model.mod4(out_2_3.to('cuda:1', non_blocking=True))
            loss_2 = crit(test_target[2:, ...], out_2_4)

            # time step 6
            loss_2.backward()

            opt.step()

            p.step()

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

def main4():
    class Lin7(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = \
                nn.Sequential(
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), nn.ReLU(),
                    nn.Linear(50, 50), 
                )
            
        def forward(self, x):
            x = self.mod1(x)
            return x
    
    mod1 = Lin7().to('cuda:0')
    mod2 = Lin7().to('cuda:1')

    mod1_comp = dynamo.optimize('inductor')(mod1).to('cuda:0')
    mod2_comp = dynamo.optimize('inductor')(mod2).to('cuda:1')
    
    class PipeModule(nn.Module):
        def __init__(self, mod1, mod2):
            super().__init__()
            self.mod1 = mod1
            self.mod2 = mod2

        def forward(self, x):
            x = x.to("cuda:0")
            x = self.mod1(x)
            x = x.to("cuda:1")
            x = self.mod2(x) # Throws an Error!
            return x
    
    model = PipeModule(mod1_comp, mod2_comp)

    opt = optim.Adam(model.parameters(), lr=1)

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


def main5():
    class PipeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = nn.Linear(50, 100, bias=False).to('cuda:0')
            self.mod2 = nn.Linear(100, 30, bias=False).to('cuda:1')
        
        def forward(self, x):
            x = self.mod1(x)
            x = x.to('cuda:1')
            x = self.mod2(x)
            return x

    model = PipeModule().train()
    opt = optim.Adam(model.parameters(), lr=1)

    # print(list(model.mod1.parameters()))
    model_name='fx_pipe'

    def print_forward(gm: torch.fx.GraphModule, example_inputs):
        print("print forward aot graph")
        draw = FxGraphDrawer(gm, model_name)
        dot = draw.get_dot_graph()
        print("get dot")
        svg = dot.create_svg()
        print("get svg")
        with open(f"{model_name}_aot_forward.svg", "wb") as f:
            f.write(svg)
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

    model = aot_module(model, fw_compiler=print_forward, bw_compiler=print_backward)
    print()
    
    opt.zero_grad()
    test_input = torch.rand(20, 50).to('cuda:0')
    test_target = torch.rand(20, 30).to('cuda:1')
    test_output = model(test_input)

    crit = nn.L1Loss()
    loss = crit(test_target, test_output)
    loss.backward()
    opt.step()

if __name__ == '__main__':
    main_hand_pipe_bfs()

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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import torch.profiler
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%y%m%d-%H%M%S")

a = torch.rand(1000, 1000).cuda()
b = torch.rand(1000, 1000).cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./result/test-{current_time}', worker_name='worker0')
) as p:
    for _ in range(10):
        c = a * b
        p.step()
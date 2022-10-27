import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.profiler
from datetime import datetime
import os

now = datetime.now()
current_time = now.strftime("%y%m%d-%H%M%S")
result_path = f'{os.getcwd()}/result/ddp-{current_time}'
cudnn.benchmark = True

dist_url = 'tcp://127.0.0.1:10101'


def run_ddp(rank, result_path):
    print(rank, result_path)
    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=1, rank=rank)

    workers = (4 + 2 - 1) // 2
    batch_size = 32 // 2

    model = models.resnet152(pretrained=True)
    
    print("model init")

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    print("model")

    device = torch.device(f"cuda:{rank}")

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainsampler = torch.utils.data.DistributedSampler(trainset, drop_last=True, shuffle=True)

    # TODO: test with pin_memory=True
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=trainsampler
    # )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=workers, sampler=trainsampler
    )

    print("clear")

    trainsampler.set_epoch(0)
    model.train()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=10,
            active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(result_path, worker_name=f'worker-{rank}'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            # TODO: set non_blocking = True
            inputs, labels = data[0].to(device=device), data[1].to(device=device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            p.step()
            if step >= 17:
                break

if __name__ == "__main__":
    print("spawn")
    mp.spawn(
        run_ddp,
        nprocs=2,
        args=tuple([result_path])
    )
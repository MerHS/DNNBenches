import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import argparse
import deepspeed

import torch.profiler
from datetime import datetime
import os

def main(args):
    now = datetime.now()
    current_time = now.strftime("%y%m%d-%H%M%S")
    result_path = f'{os.getcwd()}/result/ds-zero-{current_time}'

    cudnn.benchmark = True

    deepspeed.init_distributed()

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    model = models.resnet152(pretrained=True)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, training_data=trainset)

    fp16 = model_engine.fp16_enabled()
    print(f'fp16={fp16}')
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # device = torch.device("cuda:0")
    #model.train()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=10,
            active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(result_path, worker_name='worker0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)
            if fp16:
                inputs = inputs.half()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            p.step()
            if step >= 17:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--with_cuda', default=False, action='store_true',
                         help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                         help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                         help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                         help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    main(args)
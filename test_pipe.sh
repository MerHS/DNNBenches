#!/usr/bin/sh

python dynamo/distributed.py --torchbench_model BERT_pytorch $@
python dynamo/distributed.py --torchbench_model Background_Matting $@
python dynamo/distributed.py --torchbench_model LearningToPaint $@
python dynamo/distributed.py --torchbench_model alexnet $@
python dynamo/distributed.py --torchbench_model dcgan $@
python dynamo/distributed.py --torchbench_model densenet121 --batch_size 8 $@
python dynamo/distributed.py --torchbench_model hf_Albert $@
python dynamo/distributed.py --torchbench_model hf_Bart $@
python dynamo/distributed.py --torchbench_model hf_Bert $@
python dynamo/distributed.py --torchbench_model hf_GPT2 $@
python dynamo/distributed.py --torchbench_model hf_T5 --batch_size 4 $@
python dynamo/distributed.py --torchbench_model mnasnet1_0 $@
python dynamo/distributed.py --torchbench_model mobilenet_v2 $@
python dynamo/distributed.py --torchbench_model mobilenet_v3_large $@
python dynamo/distributed.py --torchbench_model nvidia_deeprecommender $@
python dynamo/distributed.py --torchbench_model pytorch_unet $@
python dynamo/distributed.py --torchbench_model resnet18 $@
python dynamo/distributed.py --torchbench_model resnet50 $@
python dynamo/distributed.py --torchbench_model resnext50_32x4d $@
python dynamo/distributed.py --torchbench_model shufflenet_v2_x1_0 $@
python dynamo/distributed.py --torchbench_model squeezenet1_1 $@
python dynamo/distributed.py --torchbench_model timm_nfnet $@
python dynamo/distributed.py --torchbench_model timm_efficientnet $@
python dynamo/distributed.py --torchbench_model timm_regnet $@
python dynamo/distributed.py --torchbench_model timm_resnest $@
python dynamo/distributed.py --torchbench_model timm_vision_transformer $@
python dynamo/distributed.py --torchbench_model timm_vovnet $@
python dynamo/distributed.py --torchbench_model vgg16 $@

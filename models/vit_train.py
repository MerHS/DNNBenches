from labml import experiment
from labml.configs import option
from labml_nn.transformers import TransformerConfigs

from .cifar100 import CIFAR100Configs

class Configs(CIFAR100Configs):
    """
    ## Configurations
    We use [`CIFAR10Configs`](../../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # [Transformer configurations](../configs.html#TransformerConfigs)
    # to get [transformer layer](../models.html#TransformerLayer)
    transformer: TransformerConfigs

    # Size of a patch
    patch_size: int = 4
    # Size of the hidden layer in classification head
    n_hidden_classification: int = 2048
    # Number of classes in the task
    n_classes: int = 100


@option(Configs.transformer)
def _transformer():
    """
    Create transformer configs
    """
    return TransformerConfigs()


@option(Configs.model)
def _vit(c: Configs):
    """
    ### Create model
    """
    from labml_nn.transformers.vit import VisionTransformer, LearnedPositionalEmbeddings, ClassificationHead, \
        PatchEmbeddings

    # Transformer size from [Transformer configurations](../configs.html#TransformerConfigs)
    d_model = c.transformer.d_model
    # print(c.transformer.n_layers)
    # print(c.transformer.n_heads)
    # print(c.transformer.encoder_layer.self_attn.heads)
    # Create a vision transformer
    return VisionTransformer(c.transformer.encoder_layer, c.transformer.n_layers,
                             PatchEmbeddings(d_model, c.patch_size, 3),
                             LearnedPositionalEmbeddings(d_model),
                             ClassificationHead(d_model, c.n_hidden_classification, c.n_classes)).to(c.device)


def main():
    # Create experiment
    experiment.create(name='ViT', comment='cifar100')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        # Optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,

        # Transformer embedding size
        'transformer.d_model': 512,
        # 'transformer.n_layers': 24,
        # 'transformer.n_heads': 32,

        # Training epochs and batch size
        'epochs': 32,
        'train_batch_size': 64,

        # Augment CIFAR 10 images for training
        'train_dataset': 'cifar10_train_augmented',
        # Do not augment CIFAR 10 images for validation
        'valid_dataset': 'cifar10_valid_no_augment',
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()

if __name__ == '__main__':
    main()
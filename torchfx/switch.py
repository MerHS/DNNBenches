import torch
import torch.nn as nn

from labml_nn.transformers.utils import subsequent_mask

from torch.fx import symbolic_trace
from graph_drawer import FxGraphDrawer

from node_flops import FxFlopsAdder

import torch
import torch._dynamo as dynamo

class Configs:
    heads = 4
    d_model = 128
    dropout = 0.0
    capacity_factor = 0.5
    drop_tokens = True
    is_scale_prob = True
    n_experts = 16
    d_ff = 256
    n_layers = 6
    n_tokens = 65
    device = "cuda:0"

class AutoregressiveModel(nn.Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: nn.Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        # Transformer
        self.transformer = transformer
        # Final layer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Initialize the subsequent mask
        self.mask = subsequent_mask(x.shape[0]).to(x.device)
            
        # Token embeddings
        x = self.src_embed(x)
        # Run it through the transformer
        res, counts, route_prob, n_dropped, route_prob_max = self.transformer(x, self.mask)
        # Generate logits of the next token
        res = self.generator(res)
        #
        return res, counts, route_prob, n_dropped, route_prob_max

def switch_transformer(c: Configs):
    """
    ### Initialize the switch transformer
    """
    from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
    from labml_nn.transformers import MultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward

    return SwitchTransformer(
        SwitchTransformerLayer(d_model=c.d_model,
                               attn=MultiHeadAttention(c.heads, c.d_model, c.dropout),
                               feed_forward=SwitchFeedForward(capacity_factor=c.capacity_factor,
                                                              drop_tokens=c.drop_tokens,
                                                              is_scale_prob=c.is_scale_prob,
                                                              n_experts=c.n_experts,
                                                              expert=FeedForward(c.d_model, c.d_ff, c.dropout),
                                                              d_model=c.d_model),
                               dropout_prob=c.dropout),
        c.n_layers)

def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    m = AutoregressiveModel(c.n_tokens, c.d_model, switch_transformer(c))
    return m.to(c.device)

model = autoregressive_model(Configs())
data = torch.randint(0, 64, (64, 32)).long().cuda()
target = torch.rand(64, 32).cuda()

output, counts, route_prob, n_dropped, route_prob_max = model(data)

print(output.shape)
iter_cnt =0
graph_cnt = 0
def print_compile(gm, ex):
    global iter_cnt
    global graph_cnt
    # print(
    #     f"print_compile:\n{str(gm.graph)}\n-----------------------------------------"
    # )
    graph_cnt = graph_cnt + 1
    print(f"iter: {iter_cnt}, graph: {graph_cnt}")
    
    return gm

# traced = symbolic_trace(model)

# test_input = torch.rand((16, 3, 244, 244))
# flops = FxFlopsAdder(traced)
# flops.run(test_input)

# g = FxGraphDrawer(traced, "resnet50")
# dot = g.get_dot_graph()

opt_model = dynamo.optimize(print_compile)(model)
print(opt_model(data)[0].shape)
for i in range(10):
    iter_cnt = i
    data = torch.randint(0, 64, (64, 32)).long().cuda()
    graph_cnt = 0

    print(opt_model(data)[0].shape)

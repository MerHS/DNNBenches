from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch.fx as fx

model_name = 'gpt2'

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, from_tf=False, config=config)

model.resize_token_embeddings(len(tokenizer))

module = fx.symbolic_trace(model, concrete_args={
    'input_ids': fx.PH, 'attention_mask': fx.PH, 'labels': fx.PH
})

module.graph.print_tabular()
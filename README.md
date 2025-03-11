---
base_model:
- meta-llama/Llama-3.2-1B
library_name: transformers
license: mit
pipeline_tag: text-generation
---

# TokenButler
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->



<div align="center">
  <img src="https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/tokenbutlerlogo.png?raw=true" width="50%" alt="TokenButler" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <!-- Paper Badge -->
  <a href="https://arxiv.org/abs/2503.07518" target="_blank" style="margin: 2px;">
    <img alt="Paper" 
         src="https://img.shields.io/badge/Paper-View-orange?logo=readthedocs&logoColor=white" 
         style="display: inline-block; vertical-align: middle;"/>
  </a>
  <!-- GitHub Badge -->
  <a href="https://github.com/abdelfattah-lab/TokenButler" target="_blank" style="margin: 2px;">
    <img alt="GitHub" 
         src="https://img.shields.io/badge/GitHub-Repo-black?logo=github&logoColor=white" 
         style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<br>



The collection of TokenButler models can be found [here](https://huggingface.co/collections/akhauriyash/tokenbutler-67cf181b5762d0d60e5f312b). To run the `meta-llama/Llama-3.2-1B` model, follow:

```
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

question = "If millionaires have butlers, why don't million dollar language models have a butler too? I think its because "

model_name = "akhauriyash/Llama-3.2-1B-Butler"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = generator(question, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)

print(response[0]['generated_text'][len(question):])
```

Note that the 'default' configured sparsity is 50%. Further, there is a 'sliding window' of 128 and 8 'anchor tokens'. To 'change' the sparsity, you can use the following function after loading the model. Please note that the 'fixed' is the only supported strategy at the moment, which 'fixes' the sparsity of each layer (except the first) at the 'pc' (percentage) mentioned. This can also be found at `test_hf.py`. Sliding window and anchor tokens can be changed in a similar manner.

```
def set_sparsity(model, sparsity):
    for module in model.modules():
        if module.__class__.__name__.__contains__("AttentionExperimental"):
            module.token_sparse_method = sparsity
            module.set_token_sparsity()
    return model

model = set_sparsity(model, "fixed_60pc")
```


# Predictor Architecture
<div align="center">
  <img src="https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/mainfig.png?raw=true" width="100%" alt="TokenButlerFigure" />
</div>

# Custom Synthetic Task
<div align="center">
  <img src="https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/datasetfig.png?raw=true" width="100%" alt="Synthetic Tasks" />
</div>
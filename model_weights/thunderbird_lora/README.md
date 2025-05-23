---
base_model: /home/admin1/桌面/Meta-Llama-3-8B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_Tbird
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_Tbird

This model is a fine-tuned version of [/home/admin1/桌面/Meta-Llama-3-8B-Instruct](https://huggingface.co//home/admin1/桌面/Meta-Llama-3-8B-Instruct) on the modified_Tbird_train_0-2w_Drain.json and the modified_Tbird_train_2-4w_Drain.json datasets.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.43.1
- Pytorch 2.3.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1
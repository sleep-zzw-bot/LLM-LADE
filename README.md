# LLM-LADE: Large Language Model-based Log Anomaly Detection with Explanation

This repository contains the data process code, seed data, and LoRA adapter weights for **LLM-LADE**, a framework that formulates log anomaly detection as a multi-task generation problem. It jointly predicts anomaly labels and natural language explanations using a fine-tuned LLaMA3-8B model.

> 📄 For more details, please refer to our paper:  
> **LLM-LADE: Large Language Model-based Log Anomaly Detection with Explanation**  


---

## 🔧 Dataset Preparation

1. **Download original log datasets** from [LogHub](https://github.com/logpai/loghub):

   - HDFS
   - BGL
   - Thunderbird

2. **Parse logs using the Drain log parser**. Our LoRA weights are trained on data parsed by [Drain](https://github.com/logpai/logparser).

3. **Run preprocessing** using our provided scripts:

   ```bash
   python data_process/hdfs_process.py         # For HDFS dataset
   python data_process/bgl_tbird_process.py    # For BGL and Thunderbird datasets
   ```

## 📦 LoRA Adapter Weights

We provide LoRA weights trained on three datasets:

- `model_weights/hdfs_lora/`
- `model_weights/bgl_lora/`
- `model_weights/thunderbird_lora/`

Each folder contains:

- `adapter_model.safetensors` – LoRA adapter weights
- `adapter_config.json` – LoRA configuration

These weights are trained on top of **LLaMA3-8B**, and can be used for fine-tuning or inference via PEFT or LLaMA-Factory.

------

## 🧠 Base Model Requirement

Please download the official [Meta-LLaMA3-8B model](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main) from Hugging Face and agree to its license. The LoRA weights assume this model as the base.

------

## 🚀 Inference with LLaMA-Factory

You can use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to load and evaluate the model.

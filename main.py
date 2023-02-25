# Source https://www.kaggle.com/code/julianschelb/finetune-bloom-token-classification
import transformers
from transformers import (BloomTokenizerFast,
                          BloomForTokenClassification,
                          DataCollatorForTokenClassification,
                          AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
from datasets import load_dataset
import torch
import os


model_name = "bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)
model = transformers.BloomForCausalLM.from_pretrained(f"bigscience/{model_name}")


transformers.AutoModelForCausalLM


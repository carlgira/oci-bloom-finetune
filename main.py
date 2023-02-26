# Source https://www.kaggle.com/code/julianschelb/finetune-bloom-token-classification
# https://github.com/dptrsa-300/start_with_bloom/blob/main/bloomex_nb.ipynb
# https://github.com/dredwardhyde/gpt-neo-fine-tuning-example

from transformers import (BloomTokenizerFast,
                          BloomForCausalLM,
                          TrainingArguments, Trainer, IntervalStrategy)
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

model_name = "bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)
model = BloomForCausalLM.from_pretrained(f"bigscience/{model_name}")

descriptions = pd.read_json('dataset.json')['text']
descriptions = descriptions[descriptions['text'].str.len() < 1000]['text']
max_length = max([len(tokenizer.encode(description)) for description in descriptions])


class OCIDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = OCIDataset(descriptions, tokenizer, max_length=max_length)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
training_args = TrainingArguments(output_dir='./results', num_train_epochs=1, logging_steps=100,
                                  save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                  warmup_steps=100, weight_decay=0.01, logging_dir='./logs',
                                  save_total_limit=1, load_best_model_at_end=True)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                        'attention_mask': torch.stack([f[1] for f in data]),
                                                                        'labels': torch.stack([f[0] for f in data])}).train()

generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()

'''
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',

                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
'''

import os
import sys
import torch
from torch.utils.data import Dataset


class datapre(Dataset):
    def __init__(self, jsondata, tokenizer=None):
        super(datapre, self).__init__()
        self.tokenizer = tokenizer
        self.data = [item for item in jsondata]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        enc_input = self.tokenizer(
            self.data[idx][list(self.data[idx].keys())[-2]],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc_input.input_ids.squeeze(), enc_input.attention_mask.squeeze(), \
            enc_input.token_type_ids.squeeze(), self.data[idx][list(self.data[idx].keys())[-1]]


def MyDataLoader(dataset, batch_size, shuffle=False):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=nw)
    return data_loader
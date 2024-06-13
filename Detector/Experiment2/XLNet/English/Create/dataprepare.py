import os
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader



class datapre1_complete(Dataset):
    def __init__(self, jsondata, tokenizer=None):
        super(datapre1_complete, self).__init__()
        self.tokenizer = tokenizer
        self.data = [item for item in jsondata]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        enc_input = self.tokenizer(
            self.data[idx][list(self.data[idx].keys())[-2]][-750:],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc_input.input_ids.squeeze(), enc_input.attention_mask.squeeze(), \
            enc_input.token_type_ids.squeeze(), 0


class datapre2_complete(Dataset):
    def __init__(self, jsondata, tokenizer=None):
        super(datapre2_complete, self).__init__()
        self.tokenizer = tokenizer
        self.data = [item for item in jsondata]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        enc_input = self.tokenizer(
            self.data[idx][list(self.data[idx].keys())[-1]][-750:],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc_input.input_ids.squeeze(), enc_input.attention_mask.squeeze(), \
            enc_input.token_type_ids.squeeze(), 1


def create_combined_dataloader(dataset1, dataset2, batch_size, shuffle=False):
    # 合并两个数据集
    combined_dataset = ConcatDataset([dataset1, dataset2])
    # 计算合适的线程数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 创建 DataLoader
    data_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=nw)
    return data_loader
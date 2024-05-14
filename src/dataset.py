import lightning
import torch
import datasets
import transformers
import os
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def load_wikitext(args):
    import tokenizers
    from tqdm import tqdm
    if os.path.isfile("/home/qingyu_yin/data/RWKV/WikiText-103/wiki.train.raw.pt"):
        dataset = torch.load("/home/qingyu_yin/data/RWKV/WikiText-103/wiki.train.raw.pt")
    else:
        dataset_file = "/home/qingyu_yin/data/RWKV/WikiText-103/wiki.train.raw"
        tokenizer = tokenizers.Tokenizer.from_file("src/tokenizer.json")
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = f.readlines()
        dataset = list(tokenizer.encode(
            line.strip(' ').replace('\n', '[SEP]').replace('<unk>', '[UNK]')).ids for line in tqdm(dataset))
        dataset = torch.tensor([index for line in dataset for index in line], dtype=torch.long)
        torch.save(dataset, "/home/qingyu_yin/data/RWKV/WikiText-103/wiki.train.raw.pt")
    num_sequences = (dataset.size(0) // args.ctx_len) * args.ctx_len
    dataset = dataset.narrow(0, 0, num_sequences).view(-1, args.ctx_len)
    return DataLoader(dataset, batch_size=args.micro_bsz)


class load_minipile(lightning.LightningDataModule):
    def __init__(self, args, data_type="train"):
        super().__init__()
        self.args = args
        self.data_type = data_type
        self.dataset = None

    def setup(self, stage=None):
        self.dataset = datasets.load_dataset("JeanKaddour/minipile")

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset[self.data_type], batch_size=self.args.micro_bsz, num_workers=16)
        return dataloader

    def test_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.dataset[self.data_type], batch_size=self.args.micro_bsz, num_workers=16)
        return dataloader


def load_pad_test(args,
                  zero_pad=False):
    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    tokenizer = transformers.AutoTokenizer.from_pretrained("src/gpt2")
    if zero_pad:
        def pad_to_length(example):
            example["text"] = tokenizer(example["text"], padding="max_length", max_length=args.ctx_len,)["input_ids"]


def load_pg19(args):
    dataset = datasets.load_dataset("emozilla/pg19-test", split="test")
    return DataLoader(dataset, batch_size=args.micro_bsz)


def load_superglue(args, sub_task):
    dataset = datasets.load_dataset("superglue", sub_task)
    return DataLoader(dataset["test"], batch_size=args.micro_bsz)

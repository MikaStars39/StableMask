import transformers
import datasets
import torch

from datasets import load_dataset

dataset = load_dataset("emozilla/pg19-test", split="test")
print(dataset["text"][0])


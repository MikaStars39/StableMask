import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import lightning


class ConcatenatedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=12000):
        self.tokenizer = tokenizer
        self.tokens = self._tokenize_and_concatenate(texts, max_length)

    def _tokenize_and_concatenate(self, texts, max_length):
        # 分词并拼接文本
        concatenated_tokens = []
        current_length = 0
        for text in texts:
            # 分词
            # print(text)
            tokens = tokenizer.encode(text)
            concatenated_tokens.extend(tokens)
            current_length += len(tokens)
            if current_length >= max_length:
                break
        # 保证拼接的tokens长度不超过max_length
        # print("the len is", len(concatenated_tokens))
        return concatenated_tokens[:max_length]

    def __len__(self):
        # 返回拼接后文本的长度除以512，得到不同长度的样本数量
        return len(self.tokens) // 64

    def __getitem__(self, idx):
        # 每个样本增加512个token
        start_idx = idx * 64
        end_idx = start_idx + 64 + idx * 64
        sample_length = end_idx
        # print(start_idx, idx, end_idx, len(self.tokens))
        # 从拼接的tokens中截取对应长度的片段
        sample = self.tokens[start_idx:sample_length]
        # 转换为张量
        sample_tensor = torch.tensor(sample)
        return sample_tensor


def get_args():
    from argparse import ArgumentParser
    from lightning.pytorch.trainer import Trainer
    from lightning.pytorch.utilities import rank_zero_only
    parser = ArgumentParser()

    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--decay_rate", default=0.995, type=float)
    parser.add_argument("--att_type", default="New", type=str)
    parser.add_argument("--use_model", default="conv", type=str)
    parser.add_argument("--tau", default=0.0, type=float)


    # direction & data
    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--proj_dir", default="", type=str)
    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--tokenizer_file", default="", type=str)
    parser.add_argument("--data_size", default=0, type=int)
    parser.add_argument("--test_data_file", default="", type=str)
    parser.add_argument("--piqa_type", default="valid", type=str)

    # seeds
    parser.add_argument("--random_seed", default=42, type=int)

    # dimension & size
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto
    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--n_embd", default=1024, type=int)
    parser.add_argument("--dim_att", default=1536, type=int)
    parser.add_argument("--n_layer", default=6, type=int)

    # epoch setting
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500,
                        type=int)  # train for this many "epochs". will continue afterward with lr = lr_final
    parser.add_argument("--epoch_begin", default=0,
                        type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"
    parser.add_argument("--micro_bsz", default=16, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--epoch_begin_steps", default=0, type=int)  # number of GPUs

    # optimizer
    parser.add_argument("--lr_init", default=3e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--ds_bucket_mb", default=200, type=int)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--hid_dim", type=int, default=768*2)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--precision", default="f16", type=str)

    # unknown
    parser.add_argument("--wandb", default="", type=str)
    # parser = Trainer.add_argparse_args(parser)

    # args resetting
    args = parser.parse_args()
    # create running direction

    if not os.path.exists(args.proj_dir):
        rank_zero_only(lambda: os.makedirs(args.proj_dir))
        args.default_root_dir = args.proj_dir
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.betas = (args.beta1, args.beta2)
    args.enable_checkpointing = True
    args.replace_sampler_ddp = False
    args.logger = None
    args.gradient_clip_val = 1
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = 1
    args.max_epochs = -1  # continue forever
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    args.enable_process_bar = True
    args.gradient_accumulation_steps = 1
    args.limit_train_batches = args.epoch_steps * args.gradient_accumulation_steps
    args.vocab_size = 50257
    args.hid_dim = args.n_embd * 2
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # set up running name
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"

    return args


if __name__ == "__main__":
    import wandb
    import os
    import datetime
    import torch
    import lightning
    from lightning_utilities.core.rank_zero import rank_zero_info
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    from src.trainer import train_callback

    args = get_args()

    wandblogger = WandbLogger(args.wandb)
    args.logger = wandblogger

    # seed everything
    lightning.seed_everything(args.random_seed)

    # print all keys and values of args
    rank_zero_info(str(vars(args)) + "\n")

    # set precision decision ( fp16 or fp32 ) bf16 suggested
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    from src.model import MyModule

    if args.load_model != "":
        model = MyModule.load_from_checkpoint(args.load_model, args=args)

    trainer = Trainer(accelerator="auto",
                      strategy=args.strategy,
                      num_nodes=args.num_nodes,
                      precision=args.precision,
                      logger=args.logger,
                      max_epochs=args.max_epochs,
                      min_epochs=None,
                      log_every_n_steps=args.log_every_n_steps,
                      accumulate_grad_batches=args.gradient_accumulation_steps,
                      gradient_clip_val=args.gradient_clip_val,
                      callbacks=[train_callback(args), ModelCheckpoint(dirpath=args.proj_dir, every_n_train_steps=args.epoch_save)],
                      )

    # Load the wikitext-103 dataset from "/home/qingyu_yin/data/RWKV/WikiText-103/wiki.test.raw"
    dataset = load_dataset('text', data_files='/home/qingyu_yin/data/RWKV/WikiText-103/wiki.train.raw', split='train')
    texts = dataset['text']

    # Initialize the tokenizer from tokenizer.json
    tokenizer = AutoTokenizer.from_pretrained('src/gpt2')

    # Create the dataset
    progressive_dataset = ConcatenatedTextDataset(texts, tokenizer, max_length=1638400)

    # Create a DataLoader
    dataloader = DataLoader(progressive_dataset, batch_size=1, shuffle=False)

    trainer.test(model, dataloader)

    wandb.finish()



import torch
import lightning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from src.modeling_llama import MyModule
from transformers import AutoTokenizer


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
    parser.add_argument("--random_seed", default=34, type=int)

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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hid_dim", type=int, default=768*2)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

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
    args.limit_train_batches = args.epoch_steps * args.gradient_accumulation_steps
    args.vocab_size = 50257
    args.hid_dim = args.n_embd * 2
    args.train_ctx_len = 1024
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # set up running name
    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"

    return args


args = get_args()
print("please input the path of the model you want to load")
path = input()
model = MyModule.load_from_checkpoint(path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Loading model successfully!")

tokenizer = AutoTokenizer.from_pretrained("src/gpt2")
tokenizer.model_max_length = 1024
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = True
tokenizer.padding = False


@torch.no_grad()
def generate_text(input_text):
    processed_input = tokenzier(input_text, return_tensors='pt')
    processed_input = processed_input.to(device)
    logits = model(processed_input)
    output = sample_from_logits(logits)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def sample_from_logits(logits, temperature=1.0):
    """
    从logits中采样一个token。

    :param logits: 模型的输出logits。
    :param temperature: 采样温度，用于调整概率分布的“平坦度”。
                         温度越高，输出越随机；温度越低，模型越倾向于选择最高概率的token。
    :return: 采样出的token索引。
    """
    # 应用温度调整
    logits = logits / temperature

    # 使用softmax将logits转换为概率分布
    probabilities = F.softmax(logits, dim=-1)

    # 从概率分布中随机采样
    sampled_token_id = torch.multinomial(probabilities, num_samples=1)

    return sampled_token_id


while True:
    input_text = input("Enter text to generate from (or 'quit' to exit): ")
    if input_text.lower() == 'quit':
        break
    generated_text = generate_text(input_text)
    print("Answer:", generated_text)

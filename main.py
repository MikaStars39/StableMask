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
    from datasets import load_dataset
    from src.trainer import train_callback

    args = get_args()

    if args.mode == "train" or args.mode == "test":
        wandblogger = WandbLogger(args.wandb)
        args.logger = wandblogger
    else:
        wandblogger = None

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
    model = MyModule(args)

    if args.load_model != "":
        model = MyModule.load_from_checkpoint(args.load_model, args=args)

    if args.mode == "train":
        wandblogger.watch(model)

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
                      callbacks=[train_callback(args), ModelCheckpoint(dirpath=args.proj_dir,
                                                                       every_n_train_steps=args.epoch_save,
                                                                       filename=args.att_type + '{epoch}-{step}',
                                                                       save_top_k=-1)],
                      )

    if trainer.global_rank == 0:
        # print layer info
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            elif len(shape) == 1:
                print(f"{str(shape[0]).ljust(5)}       {n}")
            else:
                print(f"{str(shape).ljust(5)}       {n}")

    if args.mode == "train":
        if "minipile" in args.data_type:
            from src.dataset import load_minipile
            dataloader = load_minipile(args)
        elif "pile" in args.data_type:
            from src.dataset import load_pile
            dataloader = load_pile(args)
        else:
            from src.dataset import load_wikitext
            dataloader = load_wikitext(args)
        trainer.fit(model, dataloader)
        wandb.finish()
    elif args.mode == "test":
        if args.data_type == "minipile":
            from src.dataset import load_minipile
            dataloader = load_minipile(args, "test")
        if args.data_type == "pg19":
            from src.dataset import load_pg19
            dataloader = load_pg19(args)
        if "superglue" in args.data_type:
            from src.dataset import load_superglue
            dataloader = load_superglue(args, args.data_type[10:])
        trainer.test(model, dataloader)

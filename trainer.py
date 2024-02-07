import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
import time


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        w_step = args.warmup_steps
        # if args.lr_final == args.lr_init or args.epoch_count == 0:
        #     lr = args.lr_init
        #     if trainer.global_step < w_step:
        #         lr = lr * (0.1 + 0.9 * trainer.global_step / w_step)
        # else:
        #     lr = args.lr_init
        #     if trainer.global_step < w_step:
        #         lr = lr * (0.1 + 0.9 * trainer.global_step / w_step)
        #     else:
        #         lr = args.lr_final + 0.5 * (args.lr_init - args.lr_final) * (
        #             1 + math.cos(math.pi * (real_step - w_step) / (args.epoch_steps * args.epoch_count - w_step))
        #         )
        #
        # for param_group in trainer.optimizers[0].param_groups:
        #     param_group["lr"] = lr
        #
        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0


    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        if trainer.is_global_zero:  # logging
            # if trainer.global_step % args.epoch_save == 0:
            #     # save the checkpoint
            #     checkpoint_path = f"{args.proj_dir}/{trainer.global_step}.ckpt"
            #     trainer.save_checkpoint(checkpoint_path)
            #     # save the weight
            #     torch.save(pl_module.state_dict(), f"{args.proj_dir}/{trainer.global_step}.pth")
            t_now = time.time_ns()
            token_per_step = args.ctx_len * args.real_bsz
            # rank_zero_info(trainer.global_step)
            real_step = trainer.global_step + args.epoch_begin_steps
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = outputs["loss"].item() * args.gradient_accumulation_steps
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count

            for param_group in trainer.optimizers[0].param_groups:
                trainer.my_lr = param_group["lr"]
                break

            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("epoch_loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_loss, prog_bar=True, on_step=True)

    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass


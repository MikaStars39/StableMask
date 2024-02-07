import math
import torch
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from src.utils import get_alibi_biases, apply_rotary_emb, RMSNorm, precompute_freqs_cis
from sec.utils import visualize_attention


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    att_type: str = "RoPE"
    max_batch_size: int = 32
    max_seq_len: int = 2048
    train_ctx_len: int = 512


def repeat_kv(x: torch.Tensor,
              n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self,
                 args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.att_type = args.att_type
        self.max_seq_len = args.max_seq_len

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if "RoPE" in self.att_type:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)
        #
        # self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv
        #
        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]
        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if "ALiBi" in self.att_type:
            if "New" in self.att_type:
                scores = scores * mask[0] + mask[1]
                scores = F.softmax(scores.float(), dim=-1).type_as(xq) * mask[0]
            else:
                scores = scores + freqs_cis + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        elif "RoPE" in self.att_type:
            if "New" in self.att_type:
                scores = scores * math.log(seqlen, 1024) * mask[0] + mask[1]
                scores = F.softmax(scores.float(), dim=-1).type_as(xq) * mask[0]
            else:
                scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        if "output_attentions" in self.att_type:
            return self.wo(output), scores.no_grad()
        else:
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self,
                 layer_id: int,
                 args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.att_type = args.att_type
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask,
    ):
        h = self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        if "output_attentions" in self.att_type:
            scores = h[1]
            h = h[0]
        h = h + x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, scores if "output_attentions" in self.att_type else out


class Transformer(nn.Module):
    def __init__(self,
                 params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.head_dim = params.dim // params.n_heads

        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,
                tokens: torch.Tensor,
                start_pos: int,):

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if "New" in self.params.att_type:
            mask = torch.full(
                (seqlen, seqlen), float("1"), device=tokens.device
            )
            mask = 1 - torch.triu(mask, diagonal=1)
            alibi_mask = get_alibi_biases(self.params.n_heads, -mask.flip(dims=[1])).flip(dims=[1]).permute(2, 0, 1)
            alibi_mask = alibi_mask.flip(dims=[0, 1]).contiguous() * (1 - mask)
            mask = [mask, alibi_mask]
        else:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            # mask = torch.hstack([
            #     torch.zeros((seqlen, start_pos), device=tokens.device),
            #     mask
            # ]).type_as(h)

        if "ALiBi" in self.params.att_type:
            mask_ = torch.full(
                (seqlen, seqlen), float("1"), device=tokens.device
            )
            mask_ = 1 - torch.triu(mask_, diagonal=1)
            alibi = get_alibi_biases(self.params.n_heads, -mask_.flip(dims=[1])).flip(dims=[1])
            freqs_cis = alibi.permute(2, 0, 1)
        else:
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        if "output_attentions" in self.params.att_type:
            scores = []
            for layer in self.layers:
                h, score = layer(h, start_pos, freqs_cis, mask)
                scores.append(score)
        else:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()

        return output, scores


class MyModule(LightningModule):
    def __init__(self,
                 args):
        super().__init__()
        self.args = args
        self.model = Transformer(ModelArgs(max_seq_len=args.ctx_len,
                                           vocab_size=args.vocab_size,
                                           dim=args.n_embd,
                                           max_batch_size=args.micro_bsz,
                                           n_layers=args.n_layer,
                                           n_heads=args.n_head,
                                           att_type=args.att_type,
                                           train_ctx_len=args.train_ctx_len,
                                           ))

        self.pad_tokenizer = AutoTokenizer.from_pretrained("src/gpt2", use_fast=True)
        self.pad_tokenizer.model_max_length = args.ctx_len + 1
        self.pad_tokenizer.pad_token = self.pad_tokenizer.eos_token
        self.pad_tokenizer.truncation = True
        self.pad_tokenizer.padding = True

    def configure_optimizers(self):
        args = self.args
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=args.lr_init,
                                      eps=args.adam_eps,
                                      betas=(args.beta1, args.beta2),
                                      weight_decay=0.1,
                                      )

        def lr_lambda(current_step):
            warmup_steps = args.warmup_steps
            total_steps = args.epoch_steps * args.epoch_count
            final_lr = args.lr_final

            if current_step < warmup_steps:
                warmup_factor = current_step / warmup_steps
                return warmup_factor
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed_lr = (1 - final_lr) * cosine_decay + final_lr
                return decayed_lr

        lr_scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        output = self.model(x, 0)
        # logits = self.LMhead(output)
        return output

    def training_step(self, batch, batch_idx):
        args = self.args

        if args.data_type == "wikitext":
            batch = batch.contiguous().to(self.device)
            idx = batch[:, :args.ctx_len - 1].to(self.device).contiguous()
            targets = batch[:, 1:args.ctx_len].to(self.device).contiguous()
        elif args.data_type == "minipile":
            batch = batch["text"]
            batch = self.pad_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            idx, mask, targets = self.mask_tensor(batch)
        elif args.data_type == "pile":
            batch = batch["text"]
            batch = self.pad_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            idx, mask, targets = self.mask_tensor(batch)
        # compute loss
        logits = self.forward(idx).contiguous()
        loss = self.compute_loss(logits, targets, mask)
        return loss


    def mask_tensor(self,
                    batch: dict, ):
        idx = batch["input_ids"].to(self.device)
        targets = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device).bool()

        idx = idx[:, :idx.size(1) - 1].contiguous()
        targets = targets[:, 1:].contiguous()
        mask = mask[:, :mask.size(1) - 1].contiguous()
        # print(targets, mask)
        return idx, mask, targets

    @staticmethod
    def compute_loss(logits: torch.Tensor,
                     targets: torch.Tensor,
                     mask: torch.Tensor = None, ):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        mask = mask.view(-1)
        sum_mask = torch.sum(mask).item()
        if sum_mask == mask.shape[0]:
            loss = torch.nn.functional.cross_entropy(logits, targets)
            # print('rank', self.global_rank, 'loss', loss.item())
        else:
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
            # loss_raw = loss
            loss = torch.sum(loss * mask) / sum_mask

        return loss

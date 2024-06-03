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
from src.utils import visualize_attention


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
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
                 hidden_dim: int,
                 n_heads: int,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.att_type = args.att_type

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask,
    ):
        bsz, seqlen, _ = x.shape
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.view(bsz, seqlen, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seqlen, self.n_heads, self.head_dim)
        values = values.view(bsz, seqlen, self.n_heads, self.head_dim)

        if "RoPE" in self.att_type:
            queries, keys = apply_rotary_emb(queries, keys, freqs_cis=freqs_cis)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if "ALiBi" in self.att_type:
            if "New" in self.att_type:
                scores = scores * mask[0] + mask[1]
                scores = F.softmax(scores.float(), dim=-1).type_as(queries) * mask[0]
            else:
                scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        elif "RoPE" in self.att_type:
            if "New" in self.att_type:
                scores = scores * math.log(seqlen, 1024) * mask[0] + mask[1]
                scores = F.softmax(scores.float(), dim=-1).type_as(queries) * mask[0]
            else:
                scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(queries)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 multiple_of: int,
                 att_type: str,
                 ):
        super().__init__()
        self.dim = hidden_dim
        self.att_type = att_type

        self.attention = Attention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
        )
        self.feed_forward = FeedForward(
            dim=hidden_dim,
            hidden_dim=4 * hidden_dim,
            multiple_of=multiple_of,
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            mask,
    ):
        h = self.attention(self.attention_norm(x), freqs_cis, mask) + x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_layers: int,
                 n_heads: int,
                 hidden_dim: int,
                 multiple_of: int,
                 max_seq_len: int,
                 norm_eps: float,
                 att_type: str,
                 ):
        super().__init__()

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                multiple_of=multiple_of,
                att_type=att_type,
            ))

        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            hidden_dim // n_heads,
            max_seq_len * 2
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_mask(self, seq_len, batch_size, att_type):
        if "New" in att_type:
                mask = torch.full(
                    (seqlen, seqlen), float("1"), device=tokens.device
                )
                mask = 1 - torch.triu(mask, diagonal=1)
                stable_mask = get_alibi_biases(self.params.n_heads, -mask.flip(dims=[1])).flip(dims=[1])
                stable_mask = stable_mask.permute(2, 0, 1)[:, 0:1, :].repeat(1, seq_len, 1)
                stable_mask = stable_mask.contiguous() * (1 - mask)
                mask = [mask, stable_mask]
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

        if "ALiBi" in att_type:
            mask_ = torch.full(
                (seqlen, seqlen), float("1"), device=tokens.device
            )
            mask_ = 1 - torch.triu(mask_, diagonal=1)
            alibi = get_alibi_biases(self.params.n_heads, -mask_.flip(dims=[1])).flip(dims=[1])
            freqs_cis = alibi.permute(2, 0, 1)
        else:
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

    def forward(self, tokens: torch.Tensor):

        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = get_mask(seqlen, _bsz, self.att_type)
        freqs_cis = self.freqs_cis[:, :seqlen]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float()

        return output


class MyModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = Transformer(
            vocab_size=args.vocab_size,
            n_layers=args.n_layer,
            n_heads=args.n_head,
            hidden_dim=args.n_embd,
            multiple_of=args.multiple_of,
            max_seq_len=args.ctx_len,
            norm_eps=args.norm_eps,
            att_type=args.att_type,
            )

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

    def predict_step(self, batch, batch_idx):
        if "super_glue" in self.args.data_type:
            passage = batch["passage"][0]
            question = batch["question"][0]
            # print(batch["label"])
            input_text = [f"Passage: {passage} Question: {question} Is this question True or False? I think it is "]
            input_text = self.pad_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            idx, mask, targets = self.mask_tensor(input_text)
            for _ in range(10):
                # print(_)
                size = idx.size(1)
                idx = torch.cat([idx, torch.zeros(1, 1024 - idx.size(1), dtype=torch.long).to(idx.device)], dim=-1)
                logits = self.forward(idx)
                logits = logits[:, :size].contiguous()
                logits = rearrange(logits, "b n d -> (b n) d")
                output = self.sample_top_p(logits, 10)
                output = rearrange(output, "(1 n) -> 1 n")
                # print(self.pad_tokenizer.decode(idx[0, :]))
                idx = idx[:, :size].contiguous()
                idx = torch.cat([idx, output[:, -1:]], dim=-1).contiguous()
                # print(self.pad_tokenizer.decode(output[0, :]))
            print(self.pad_tokenizer.decode(idx[0, :]))

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
        else:
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
            loss = torch.sum(loss * mask) / sum_mask

        return loss

    @staticmethod
    def sample_top_p(probs, p, temperature=0.5):
        probs = F.softmax(probs / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token[:, 0]


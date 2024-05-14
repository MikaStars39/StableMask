# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
import os
import io
import wandb
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from src.utils import get_alibi_biases


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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.att_type = args.att_type

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
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
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
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3))
        if "ALiBi" in self.att_type:
            if "New" in self.att_type:
                scores = scores * mask + freqs_cis
                scores = F.softmax(scores.float(), dim=-1).type_as(xq) * mask
            else:
                scores = scores + freqs_cis
        elif "New" in self.att_type:
            scores = scores / math.sqrt(self.head_dim)
            scores = scores * mask[0] + mask[1]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) * mask[0]
            # print(scores)
        elif "one" in self.att_type:
            scores = scores / math.sqrt(self.head_dim)
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            zero_score = torch.zeros_like(scores)
            scores = torch.cat([zero_score[:, :, :1, :], scores], dim=-2)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = scores[:, :, 1:, :].contiguous()
        else:
            scores = scores / math.sqrt(self.head_dim)
            scores = scores + mask
            # zero_score = torch.zeros_like(scores)
            # scores = torch.cat([zero_score[:, :, :1, :], scores], dim=-2)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # scores = scores[:, :, 1:, :].contiguous()
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output), scores.no_grad()


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
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
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h, scores = self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        h = h + x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, scores


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
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
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, tokens: torch.Tensor,
                start_pos: int,
                ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if "New" in self.params.att_type:
            mask = torch.full(
                (seqlen, seqlen), float("1"), device=tokens.device
            )
            mask = 1 - torch.triu(mask, diagonal=1)
            if "RoPE" in self.params.att_type:
                alibi_mask = get_alibi_biases(self.params.n_heads, -mask.flip(dims=[1])).flip(dims=[1])
                alibi_mask = alibi_mask.permute(2, 0, 1)
                alibi_mask = alibi_mask + torch.triu(alibi_mask.flip(dims=[1, 2]), diagonal=1)
                alibi_mask = alibi_mask * (1 - mask) + torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=tokens.device),
                                                                  diagonal=self.params.train_ctx_len).flip(dims=[0, 1])
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
            alibi = get_alibi_biases(self.params.n_heads, -mask.flip(dims=[1])).flip(dims=[1])
            freqs_cis = alibi.permute(2, 0, 1)
            freqs_cis = freqs_cis + torch.triu(freqs_cis.flip(dims=[1, 2]), diagonal=1)
        else:
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        for layer in self.layers:
            h, scores = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class MyModule(LightningModule):
    def __init__(self, args):
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
        # self.LMhead = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self.pad_tokenizer = AutoTokenizer.from_pretrained("src/gpt2", use_fast=True)
        # set the max length
        self.pad_tokenizer.model_max_length = args.ctx_len + 1
        # set the padding token
        self.pad_tokenizer.pad_token = self.pad_tokenizer.eos_token
        # set the truncation
        self.pad_tokenizer.truncation = True
        # set the padding
        self.pad_tokenizer.padding = True

    def configure_optimizers(self):
        args = self.args
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=args.lr_init,
                                      eps=args.adam_eps,
                                      betas=(args.beta1, args.beta2),
                                      weight_decay=0.001,
                                      )

        def lr_lambda(current_step):
            warmup_steps = args.warmup_steps
            total_steps = args.epoch_steps * args.epoch_count
            final_lr = args.lr_final

            if current_step < warmup_steps:
                warmup_factor = (current_step / warmup_steps)
                return warmup_factor
            else:
                decay_factor = (current_step - warmup_steps) / (total_steps - warmup_steps)
                return (1 - decay_factor) * (1 - final_lr) + final_lr

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
            batch = self.pad_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            idx, mask, targets = self.mask_tensor(batch)
        # compute loss
        logits = self.forward(idx)

        # test if the variable mask exists
        if logits.size(1) > targets.size(1):
            logits = logits[:, :targets.size(1), :]
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
        elif logits.size(1) < targets.size(1):
            targets = targets[:, :logits.size(1)]
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss

    def test_step(self, batch, batch_idx):
        args = self.args

        if args.data_type == "wikitext":
            batch = batch.contiguous().to(self.device)
            idx = batch[:, :args.ctx_len - 1].to(self.device).contiguous()
            targets = batch[:, 1:args.ctx_len].to(self.device).contiguous()
        elif args.data_type == "minipile":
            batch = batch["text"]
            batch = self.pad_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            idx, mask, targets = self.mask_tensor(batch)
        elif args.data_type == "pg19":
            batch = batch["text"]
            for i in range(len(batch)):
                batch[i] = " " * 412 + batch[i][:100]
            self.pad_tokenizer.padding_side = "right"
            batch = self.pad_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            idx, mask, targets = self.mask_tensor(batch)
            print(idx.shape)
        else:
            raise ValueError("data type not supported")
        # compute loss

        # logits = self.forward(idx[:, :512])
        # for i in range(513, idx.size(1)):
        #     logits_seg = self.forward(idx[:, i-512:i])
        #     logits = torch.cat([logits, logits_seg[:, 511:, :]], dim=1).contiguous()
        #     loss = self.compute_loss(logits, targets[:, :logits.size(1)].contiguous(), mask[:, :logits.size(1)].contiguous())
        #     if i % 500 == 0:
        #         print(loss.item())
        #
        #     del logits_seg
        logits = self.forward(idx)
        logits = logits[:, 300:, :].contiguous()
        targets = targets[:, 300:].contiguous()
        loss = self.compute_loss(logits, targets, mask)
        print(loss)
        accuracy = self.compute_accuracy(logits, targets, mask)
        print(accuracy)
        self.log('test_loss', loss, batch_size=targets.size(0), on_step=True, on_epoch=True, prog_bar=True)

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
                     mask: torch.Tensor = None,):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        mask = mask.view(-1)
        loss = torch.nn.functional.cross_entropy(logits, targets)


        return loss

    def compute_accuracy(self, logits: torch.Tensor,
                     targets: torch.Tensor,
                     mask: torch.Tensor = None,):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        mask = mask.view(-1)
        logits = torch.argmax(logits, dim=-1)
        correct = torch.sum(logits == targets)
        total = torch.sum(mask)
        return correct / total


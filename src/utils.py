import torch
import matplotlib.pyplot as plt
import math
from typing import Tuple


def visualize_attention(attention_list):
    """
    Visualizes attention matrices for each head in each layer.

    :param attention_list: List of attention matrices from transformer layers
                           Each matrix should be of the shape (1, num_heads, seq_len, seq_len)
    """
    for layer_num, attention_matrix in enumerate(attention_list):
        num_heads = attention_matrix.size(1)

        # Ensure the attention matrix is on CPU and detach it from the computation graph
        attention_matrix = attention_matrix.cpu().detach()

        for head in range(num_heads):
            plt.figure(figsize=(10, 10))
            plt.matshow(attention_matrix[0, head], cmap='Blues')
            plt.title(f"Layer {layer_num + 1} Head {head + 1}", pad=20)

            # Save the figure
            plt.savefig(f"Layer_{layer_num + 1}_Head_{head + 1}.png")
            plt.close()

def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    return m


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """

    # Get slopes $m$ for each head
    m = get_slopes(n_heads).to(mask.device)

    # Calculate distances $[0, 1, \dots, N]$
    # Here we calculate the distances using the mask.
    #
    # Since it's causal mask we can just use $[0, 1, \dots, N]$ too.
    # `distance = torch.arange(mask.shape[1], dtype=torch.long, device=mask.device)[None, :]`
    distance = mask.cumsum(dim=-1)

    # Multiply them pair-wise to get the AliBi bias matrix
    return distance[:, :, None] * m[None, None, :]


class RMSNorm(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 eps: float = 1e-6):
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


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0):
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


def reshape_for_broadcast(freqs_cis: torch.Tensor,
                          x: torch.Tensor):
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

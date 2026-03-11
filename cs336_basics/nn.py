from typing import Callable
from jaxtyping import Float, Integer
import torch
from torch import Tensor, nn
import einx
from einx import dot, get_at, multiply


def linear_weight_init(weight: torch.Tensor) -> None:
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    sigma = (2 / (in_features + out_features)) ** 0.5
    nn.init.trunc_normal_(weight, mean=0.0,
        std=sigma,
        a=-3.0 * sigma, b=3.0 * sigma)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features),
            device=device, dtype=dtype))
        linear_weight_init(self.weight)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return dot("... [d_in], d_out [d_in] -> ... d_out", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim),
            device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0,
            std=1.0,
            a=-3.0, b=3.0)

    def forward(self, token_ids: Integer[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return get_at("[vocab_size] d_model, ... -> ... d_model", self.weight, token_ids)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / ((x.pow(2) + self.eps).mean(dim=-1, keepdim=True) ** 0.5) * self.weight
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def glu(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid
) -> torch.Tensor:
    act = activation(dot("... [d_model], d_ff [d_model] -> ... d_ff", x, w1))
    return act * dot("... [d_model], d_ff [d_model] -> ... d_ff", x, w2)


def swiglu(x: torch.Tensor, w1_weight: torch.Tensor, w2_weight: torch.Tensor,
    w3_weight: torch.Tensor
) -> torch.Tensor:
    # W2 * (SiLU(W1 @ x) * W3)
    return dot("... [d_ff], d_model [d_ff] -> ... d_model", glu(x, w1_weight, w3_weight, silu), w2_weight)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer.

    Args:
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward layer.
        device (torch.device | None, optional): Device to use. Defaults to None.
        dtype (torch.dtype | None, optional): Data type to use. Defaults to None.

    Returns:
        Float[Tensor, "... d_model"]: Output tensor of the same shape as the input tensor.

    Note:
        You should set d_ff to approximately a multiple of 64 that closes to 8 / 3 * d_model
            to make good use of the GPU.
    """
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        linear_weight_init(self.w1_weight)
        linear_weight_init(self.w2_weight)
        linear_weight_init(self.w3_weight)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return swiglu(x, self.w1_weight, self.w2_weight, self.w3_weight)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding.

    Args:
        theta (float): The theta parameter for the RoPE.
        d_k (int): The dimension of the key and value vectors.
        max_seq_len (int): The maximum sequence length.
        device (torch.device | None, optional): Device to use. Defaults to None.

    References:
        - https://kexue.fm/archives/8265
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # [d_k // 2]
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k)))

    def rotate_half(self, x: Float[Tensor, " ... seq_len d_k"]) -> Float[Tensor, " ... seq_len d_k"]:
        assert x.shape[-1] == self.d_k
        x[..., 1::2] *= -1
        x = torch.cat((x[..., 1::2], x[..., 0::2]), dim=-1)
        x = x.reshape(-1, self.d_k)
        return x

    def forward(self, x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Integer[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len"]:
        seq_len = x.shape[-2]
        assert x.shape[-1] == self.d_k
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        freqs = multiply("... seq_len, d_2 -> ... seq_len d_2", token_positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        # x_0 -> x_0 * cos(theta) - x_1 * sin(theta)
        # x_1 -> x_1 * cos(theta) + x_0 * sin(theta)
        return x * cos + self.rotate_half(x) * sin

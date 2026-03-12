import math
from typing import Callable, Iterable, Tuple
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
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)


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
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        # Precompute cos/sin for positions [0, max_seq_len): (max_seq_len, d_k//2) -> (max_seq_len, d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = multiply("max_seq_len, dk_2 -> max_seq_len dk_2", positions, inv_freq)
        cos_cache = freqs.cos().repeat_interleave(2, dim=-1)   # (max_seq_len, d_k)
        sin_cache = freqs.sin().repeat_interleave(2, dim=-1)   # (max_seq_len, d_k)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def rotate_half(self, x: Float[Tensor, " ... seq_len d_k"]) -> Float[Tensor, " ... seq_len d_k"]:
        assert x.shape[-1] == self.d_k
        # Reshape to pairs (x_0, x_1), (x_2, x_3), ... then (-x_1, x_0), (-x_3, x_2), ...
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)
        return rotated.reshape(*x.shape[:-1], -1)

    def forward(self, x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Integer[Tensor, " ... seq_len"] | None = None
    ) -> Float[Tensor, " ... seq_len d_k"]:
        seq_len = x.shape[-2]
        assert x.shape[-1] == self.d_k
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        # Index precomputed cos/sin by token_positions: (..., seq_len, d_k)
        if token_positions is None:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            cos = get_at("[max_seq_len] d_k, ... seq_len -> ... seq_len d_k", self.cos_cache, token_positions)
            sin = get_at("[max_seq_len] d_k, ... seq_len -> ... seq_len d_k", self.sin_cache, token_positions)
        # x_0 -> x_0 * cos(theta) - x_1 * sin(theta); x_1 -> x_1 * cos(theta) + x_0 * sin(theta)
        return x * cos + self.rotate_half(x) * sin


def softmax(x: Float[Tensor, " ... d_model"], dim: int = -1) -> Float[Tensor, " ... d_model"]:
    m = x.max(dim=dim, keepdim=True).values
    # NOTE: this is numerically stable
    e = torch.exp(x - m)
    s = e.sum(dim=dim, keepdim=True)
    return e / s


def scaled_dot_product_attention(query: Float[Tensor, "batch_size ... seq_len d_k"],
    key: Float[Tensor, "batch_size ... seq_len d_k"],
    value: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Float[Tensor, "... seq_len seq_len"] | None = None
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k = query.shape[-1]
    scores = dot("batch_size ... query_len [d_k], batch_size ... key_len [d_k] "
        "-> batch_size ... query_len key_len", query, key)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = softmax(scores, dim=-1)
    return dot("batch_size ... query_len [key_len], batch_size ... [key_len] d_v "
        "-> batch_size ... query_len d_v", scores, value)


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self,
        x: Float[Tensor, "batch_size seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Integer[Tensor, " ... seq_len"] | None = None,
    )-> Float[Tensor, "batch_size seq_len d_model"]:
        mask = torch.logical_not(torch.triu(torch.ones((x.shape[-2], x.shape[-2]), device=x.device,
            dtype=torch.bool), diagonal=1))
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.d_k)
        q = self.q_proj(x).view(*hidden_shape).transpose(1, 2)
        k = self.k_proj(x).view(*hidden_shape).transpose(1, 2)
        v = self.v_proj(x).view(*hidden_shape).transpose(1, 2)
        if rope is not None:
            q = rope(q, token_positions)
            k = rope(k, token_positions)
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = attn.transpose(1, 2).reshape(*input_shape, self.d_model)
        return self.output_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
        theta: float, max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = CausalMultiHeadAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)

    def forward(self,
        x: Float[Tensor, "batch_size seq_len d_model"]
    )-> Float[Tensor, "batch_size seq_len d_model"]:
        x = x + self.attn(self.ln1(x), self.rope)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
        theta: float, context_length: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.layers = nn.ModuleList([TransformerBlock(
            d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype
        ) for _ in range(num_layers)])
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self,
        x: Integer[Tensor, "batch_size seq_len"],
    )-> Float[Tensor, "batch_size seq_len vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


def log_softmax(x: Float[Tensor, " ... d_model"], dim: int = -1) -> Float[Tensor, " ... d_model"]:
    # Numerically stable: subtract max to avoid overflow
    m = x.max(dim=dim, keepdim=True).values
    return x - m - torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))


def cross_entropy(
    input_logits: Float[Tensor, " ... vocab_size"],
    target_ids: Integer[Tensor, " ... "]
) -> Float[Tensor, ""]:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    References:
        - https://jaykmody.com/blog/stable-softmax/
    """
    log_probs = log_softmax(input_logits, dim=-1)
    score = get_at("... [vocab_size], ... -> ...", log_probs, target_ids)
    loss = -torch.mean(score)
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                alpha_t = lr * (1 - betas[1] ** t) ** 0.5 / (1 - betas[0] ** t)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                # Apply weight decay
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float, eps: float = 1e-8
) -> None:
    l2_norm = torch.linalg.vector_norm(torch.stack(
        [p.grad.data for p in parameters if p.grad is not None], dim=0), ord=2)
    clip_coef = max_l2_norm / (l2_norm + eps)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad.data *= clip_coef_clamped


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    t_w: int,
    t_c: int
) -> float:
    """
    Get the learning rate for the given iteration.
    Args:
        t: The current iteration.
        alpha_max: The maximum learning rate.
        alpha_min: The minimum learning rate.
        t_w: The warmup period.
        t_c: The cosine cycle period.
    Returns:
        The learning rate for the given iteration.
    """
    if t < t_w:
        return alpha_max * t / t_w
    elif t < t_c:
        return alpha_min + (alpha_max - alpha_min) * \
            (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) / 2
    else:
        return alpha_min
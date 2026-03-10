import torch
from torch import Tensor, nn
from einx import dot, get_at

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features),
            device=device, dtype=dtype))
        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0,
            std=sigma,
            a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dot("... d_in, d_out d_in -> ... d_out", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim),
            device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0,
            std=1.0,
            a=-3.0, b=3.0)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return get_at("[vocab_size] d_model, ... -> ... d_model", self.weight, token_ids)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / ((x.pow(2) + self.eps).mean(dim=-1, keepdim=True) ** 0.5) * self.weight
        return result.to(in_dtype)
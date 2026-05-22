import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from .base import BaseModel
from .registry import MODEL_REGISTRY


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (three h d) -> three b h l d",
                             three=3, h=self.num_heads).unbind(0)

        if self.use_flash:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = self.dropout(F.softmax(attn, dim=-1))
            x = attn @ v

        x = rearrange(x, "b h l d -> b l (h d)")
        return self.out_proj(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads,
                                        attention_dropout, use_flash)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


@MODEL_REGISTRY.register("transformer")
class TransformerModel(BaseModel):
    """Generic Transformer model for sequence tasks."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                use_flash=cfg.use_flash_attention,
            )
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        mask = batch.get("attention_mask", None)
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.embedding(input_ids) + self.pos_embedding(positions))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.head(x)

        return {"logits": logits, "hidden_states": x}

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"][:, :-1].contiguous()
        targets = batch["input_ids"][:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return {"loss": loss}

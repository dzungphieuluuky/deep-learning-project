import torch
import torch.nn as nn
from einops import rearrange, reduce
from transformers import PreTrainedModel, PretrainedConfig

class SimpleEinopsNet(nn.Module):
    """
    A sample network demonstrating Einops and standard PyTorch layers.
    Can be replaced by a HF Transformer or Diffuser model.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        :param input_dim: Flattened input dimension.
        :param hidden_dim: Hidden layer dimension.
        :param output_dim: Number of classes.
        """
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (Batch, Channels, Height, Width)
        :return: Logits
        """
        # Example Einops: Flatten image spatial dimensions
        # 'b c h w -> b (c h w)'
        if x.ndim == 4:
            x = rearrange(x, 'b c h w -> b (c h w)')
            
        return self.block(x)
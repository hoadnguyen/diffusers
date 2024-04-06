import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, batch_first=True):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim**2, num_heads=num_heads, batch_first=batch_first
        )

    def forward(self, query, key_value) -> torch.Tensor:
        q = query.flatten(-2, -1)
        kv = key_value.flatten(-2, -1)
        output, _ = self.multihead_attn(q, kv, kv)
        batch_size = q.size(0)
        return output.view(batch_size, -1, self.embed_dim, self.embed_dim)

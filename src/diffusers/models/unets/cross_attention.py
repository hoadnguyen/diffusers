import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, head_dim=1, num_heads=4):
        super().__init__()
        
        assert query_dim == key_value_dim, "Query and key/value dimensions must match"
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = head_dim * num_heads

        self.query_projection = nn.Linear(query_dim, self.hidden_dim)
        self.key_projection = nn.Linear(key_value_dim, self.hidden_dim)
        self.value_projection = nn.Linear(key_value_dim, self.hidden_dim)
        self.final_projection = nn.Linear(self.hidden_dim, query_dim)
        self.batch_norm = nn.BatchNorm1d(query_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key_value):
        batch_size = query.size(0)

        # Project and split the inputs into multiple heads
        q = self.query_projection(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_projection(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_projection(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate the attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(q.device)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the attention weights to the values
        output = torch.matmul(attention_weights, v)

        # Concatenate the heads together
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Final projection layer
        output = self.final_projection(output)
        
        output = self.batch_norm(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output

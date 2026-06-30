import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, mask=None):
        batch_size, num_queries, _ = query.size()
        num_keys = key.size(1)

        q = self.q_proj(query).reshape(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, num_keys, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, num_queries, self.dim)
        return self.out_proj(attn_output)


class SubgraphAwareAttention(nn.Module):
    def __init__(self, num_subgraphs = 8, embed_dim=256):
        super().__init__()
        self.num_subgraphs = num_subgraphs
        self.embed_dim = embed_dim
        self.modal_cuts = [0, 23, 59, 82, 101, 127, 144, 169, 200]
        self.num_bottlenecks = 2
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = nn.Dropout(0.1)
        self.FFN_subgraph = True

        self.bottlenecks = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.num_bottlenecks, embed_dim))
            for _ in range(num_subgraphs)
        ])

        self.attn_layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadCrossAttention(embed_dim, self.num_heads)
                for _ in range(num_subgraphs)
            ])
            for _ in range(self.num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(embed_dim)
                for _ in range(num_subgraphs)
            ])
            for _ in range(self.num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(self.dropout_rate),
                )
                for _ in range(num_subgraphs)
            ])
            for _ in range(self.num_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(embed_dim)
                for _ in range(num_subgraphs)
            ])
            for _ in range(self.num_layers)
        ])

    def forward(self, inputs, batch=None):
        batch_size = inputs.size(0)
        modality_embeddings = []
        modality_bottlenecks = []

        for i in range(self.num_subgraphs):
            start, end = self.modal_cuts[i], self.modal_cuts[i + 1]
            modality_embeddings.append(inputs[:, start:end, :])
            modality_bottlenecks.append(self.bottlenecks[i].expand(batch_size, -1, -1))

        for layer_idx in range(self.num_layers):
            new_embeddings = []
            new_bottlenecks = []

            for i in range(self.num_subgraphs):
                query = torch.cat([modality_embeddings[i], modality_bottlenecks[i]], dim=1)
                key_value = torch.cat([
                    torch.cat([modality_embeddings[j], modality_bottlenecks[j]], dim=1)
                    for j in range(self.num_subgraphs)
                ], dim=1)

                if self.FFN_subgraph:
                    attn_out = self.attn_layers[layer_idx][i](query, key_value, key_value)
                    fused = self.norms[layer_idx][i](query + self.dropout(attn_out))
                    ffn_out = self.ffn_layers[layer_idx][i](fused)
                    fused = self.ffn_norms[layer_idx][i](fused + ffn_out)
                else:
                    fused = self.attn_layers[layer_idx][i](query, key_value, key_value)
                    fused = self.norms[layer_idx][i](query + fused)

                node_count = modality_embeddings[i].size(1)
                new_embeddings.append(fused[:, :node_count, :])
                new_bottlenecks.append(fused[:, node_count:, :])

            modality_embeddings = new_embeddings
            modality_bottlenecks = new_bottlenecks

        return modality_embeddings

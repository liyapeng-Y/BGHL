import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, edge_index):
        h = torch.matmul(x, self.W)
        row, col = edge_index

        h_i = h[row]
        h_j = h[col]
        a_input = torch.cat([h_i, h_j], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))

        attention = torch.exp(e)
        attention_sum = torch.zeros(h.size(0), device=x.device)
        attention_sum = attention_sum.index_add(0, row, attention)
        attention = attention / (attention_sum[row] + 1e-16)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.zeros_like(h)
        h_prime = h_prime.index_add(0, row, attention.unsqueeze(1) * h_j)

        if self.concat:
            return F.elu(h_prime)
        return h_prime

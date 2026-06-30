# -*- coding: utf-8 -*-
import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from . import lorentz as L
from .Gamamba import Gamamba
from .SubgraphAwareAttention import SubgraphAwareAttention
from .ptdec import DEC


class GraphMambaLayer(nn.Module):
    def __init__(self, dim_h, chunk_size=50, dropout=0.0, dim_state=16, dim_conv=4, expand=1):
        super().__init__()

        mamba_args = SimpleNamespace(
            d_model=dim_h,
            d_inner=dim_h * 2,
            d_state=dim_state,
            d_conv=dim_conv,
            expand=expand,
            nheads=1,
            headdim=dim_h * 2,
            chunk_size=chunk_size,
        )
        self.mamba = Gamamba(use_gnn=True, args=mamba_args, device="cuda")

        self.norm = nn.LayerNorm(dim_h)
        self.dropout = nn.Dropout(dropout)
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.ff_norm = nn.LayerNorm(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h_attn, _ = self.mamba(x, edge_index)
        h_attn = self.norm(x + self.dropout(h_attn))
        h = h_attn + self._ff_block(h_attn)
        return self.ff_norm(h)

    def _ff_block(self, x):
        x = self.ff_dropout1(torch.relu(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


class GraphMambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphMambaLayer(d_model, chunk_size=50, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, batch):
        output = None
        for layer in self.layers:
            h_dense, _ = to_dense_batch(x, batch)
            output = layer(h_dense, edge_index)
        return self.norm(output)


class SubGraphMambaEncoder(nn.Module):
    def __init__(self, d_model, num_layers=2, dropout=0.0):
        super().__init__()
        self.comm_dims = [23, 36, 23, 19, 26, 17, 25, 31]
        self.use_residual = True
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                f"comm_{i}": GraphMambaLayer(d_model, chunk_size=dim, dropout=dropout)
                for i, dim in enumerate(self.comm_dims)
            })
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, subgraphs_list, edge_index_list):
        outputs = subgraphs_list
        for layer in self.layers:
            layer_outputs = []
            for i, output in enumerate(outputs):
                transformed = layer[f"comm_{i}"](output, edge_index_list[i])
                if self.use_residual:
                    transformed = transformed + output
                layer_outputs.append(self.norm(transformed))
            outputs = layer_outputs
        return torch.cat(outputs, dim=1)


class HLBG(nn.Module):
    def __init__(
        self,
        in_size,
        num_class,
        d_model,
        dropout=0.0,
        num_layers=4,
        pe=False,
        pe_dim=0,
        **kwargs,
    ):
        super().__init__()

        curv_init = kwargs.get("curv_init", 1.0)
        learn_curv = kwargs.get("learn_curv", False)
        self.curv_raw = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.entail_weight = kwargs.get("entail_weight", 0.2)
        self.entail_weight_1 = kwargs.get("entail_weight_1", 0.2)
        self.weight_att = kwargs.get("weight_att", 0.2)

        self.global_alpha_raw = nn.Parameter(torch.tensor(d_model**-0.5).log())
        self.local_alpha_raw = nn.Parameter(torch.tensor(d_model**-0.5).log())
        self.reduce_dim = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
        )

        self.pe = pe
        self.pe_dim = pe_dim
        if pe and pe_dim > 0:
            self.embedding_pe = nn.Linear(pe_dim, d_model)

        self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=True)
        self.embedding1 = nn.Linear(in_features=in_size, out_features=d_model, bias=True)
        self.node_rearranged_len = [0, 23, 59, 82, 101, 127, 144, 169, 200]

        self.gt_1_encoder = GraphMambaEncoder(d_model=d_model, num_layers=num_layers, dropout=dropout)
        self.gt_encoder = SubGraphMambaEncoder(d_model=d_model, num_layers=num_layers, dropout=dropout)

        self.encoder = nn.Sequential(
            nn.Linear(d_model * 200 * 2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, d_model * 200 * 2),
        )
        self.dec_1 = DEC(
            cluster_number=100,
            hidden_dimension=d_model * 2,
            encoder=self.encoder,
            orthogonal=True,
            freeze_center=False,
            project_assignment=True,
        )
        self.dim_reduction = nn.Sequential(
            nn.Linear(d_model * 2, 8),
            nn.LeakyReLU(),
        )
        self.fc_local = nn.Sequential(
            nn.Linear(100 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_class),
        )
        self.fusion = SubgraphAwareAttention()

    
        self.attention_model = HyperAGG(256, 4)
        self.attention_model_global = HyperAGG(256, 4)
        self.self_attention = nn.MultiheadAttention(d_model * 2, num_heads=2, batch_first=True)

    def collect_subgraph_edge_indices(self, data_list, num_subgraphs=8):
        edge_index_batches = []

        for k in range(num_subgraphs):
            offset = 0
            all_edge_index = []

            for data in data_list:
                x = getattr(data, f"comm_nodes_{k}")
                edge_index = getattr(data, f"subgraph_edge_index_{k}")

                all_edge_index.append(edge_index + offset)
                offset += x.size(0)

            edge_index_batches.append(torch.cat(all_edge_index, dim=1))

        return edge_index_batches

    def split_output_by_comm(self, output, batch):
        output, _ = to_dense_batch(output, batch)
        comm_dims = [
            self.node_rearranged_len[i + 1] - self.node_rearranged_len[i]
            for i in range(len(self.node_rearranged_len) - 1)
        ]
        return list(torch.split(output, comm_dims, dim=1))

    def _euclid_to_lorentz(self, x_euclid: torch.Tensor, scale: torch.Tensor, K) -> torch.Tensor:
        x_euclid = self.reduce_dim(x_euclid)
        x_scaled = x_euclid * scale.exp()
        zeros = torch.zeros_like(x_scaled[..., :1])
        x_tan0 = torch.cat([zeros, x_scaled], dim=-1)
        return L.exp_map0(x_tan0, K)

    def compute_entail_loss_batched(self, x, y, curv, alpha: float = 1.0, reduction: str = "mean"):
        assert x.dim() == 3 and y.dim() == 3, f"Expect x,y as [B, M, N] and [B, K, N], got {x.shape}, {y.shape}"
        assert x.shape[0] == y.shape[0], "batch size must match"
        assert x.shape[-1] == y.shape[-1], "feature dim must match"

        angle = L.oxy_angle(x.unsqueeze(2), y.unsqueeze(1), curv)
        apert = L.half_aperture(x, curv).unsqueeze(-1)
        factor = ((angle / apert - 1).clamp(max=3)).exp()
        loss_map = factor * (angle - alpha * apert).clamp(min=0.0)

        if reduction == "none":
            return loss_map
        if reduction == "mean":
            return loss_map.mean()
        if reduction == "sum":
            return loss_map.sum()
        raise ValueError(f"Unknown reduction: {reduction}")

    def forward(self, data, train, return_attn=False):
        x, edge_index = data.x, data.edge_index
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN.")

        batch = data.batch
        pe = data.pe if hasattr(data, "pe") else None
        coord = data.coord if hasattr(data, "coord") else None
        edge_index_subgraphs = self.collect_subgraph_edge_indices(data.to_data_list())

        output_global = self.embedding(x)
        output_local = self.embedding1(x)

        if self.pe and pe is not None:
            if coord is not None:
                coord = coord / torch.norm(coord, dim=1, keepdim=True)
                pe = torch.cat([pe, coord], dim=-1)
            pe = self.embedding_pe(pe)
            output_global = output_global + pe
            output_local = output_local + pe

        output_local = self.split_output_by_comm(output_local, batch)
        output_local = self.gt_encoder(output_local, edge_index_subgraphs)
        output_global = self.gt_1_encoder(output_global, edge_index, batch)

        output_local = self.fusion(output_local, batch)

        curv_log_min = self._curv_minmax["min"]
        curv_log_max = self._curv_minmax["max"]
        curv_log = curv_log_min + (curv_log_max - curv_log_min) * torch.sigmoid(self.curv_raw)
        curv = curv_log.exp()

        output_global_parent = self.attention_model_global(torch.cat(output_local, dim=1))
        output_global_hyperbolic = self._euclid_to_lorentz(output_global_parent, self.global_alpha_raw, curv)

        parent_node_list = []
        entailment_loss = 0
        for local_output in output_local:
            parent_node = self.attention_model(local_output)
            local_hyperbolic = self._euclid_to_lorentz(local_output, self.global_alpha_raw, curv)
            parent_hyperbolic = self._euclid_to_lorentz(parent_node, self.global_alpha_raw, curv)
            parent_node_list.append(parent_hyperbolic)
            entailment_loss += self.compute_entail_loss_batched(
                parent_hyperbolic,
                local_hyperbolic,
                curv,
                reduction="mean",
            )
        entailment_loss = entailment_loss / len(output_local)

        entailment_loss_1 = self.compute_entail_loss_batched(
            output_global_hyperbolic,
            torch.cat(parent_node_list, dim=1),
            curv,
            reduction="mean",
        )

        output_local = torch.cat(output_local, dim=1)
        output_global = torch.cat([output_global, output_local], dim=2)
        attended, _ = self.self_attention(output_global, output_global, output_global)
        output_global = output_global + self.weight_att * attended

        x_local, _ = self.dec_1(output_global.contiguous())
        x_local = self.dim_reduction(x_local)
        x_local = x_local.reshape((x_local.shape[0], -1))
        locals_logits = self.fc_local(x_local)

        return locals_logits, self.entail_weight * entailment_loss + self.entail_weight_1 * entailment_loss_1


class HyperAGG(nn.Module):
    def __init__(self, input_dim, reduce, act="gelu", dropout=0.25):
        super().__init__()
        hidden_dim = 256
        attention_dim = hidden_dim // reduce

        if act.lower() == "gelu":
            feature = [
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ]
        else:
            feature = [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ]

        if dropout:
            feature.append(nn.Dropout(dropout))

        self.feature = nn.Sequential(*feature)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.apply(initialize_weights)

    def forward(self, x):
        feature = self.feature(x)
        attention = self.attention(feature).transpose(-1, -2)
        attention = F.softmax(attention, dim=-1)
        return torch.bmm(attention, feature)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

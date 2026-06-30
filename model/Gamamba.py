from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from .gnn import GraphAttentionLayer

Device = Union[str, torch.device, None]


class Gamamba(nn.Module):
    def __init__(self, use_gnn, args, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device
        self.use_gnn = use_gnn

        if self.use_gnn:
            self.gnn = GraphAttentionLayer(
                in_features=self.args.d_model * 2,
                out_features=self.args.d_state,
            )
            self.projector = nn.Linear(self.args.d_state, self.args.d_state)

        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)
        self.gnn_norm = RMSNorm(self.args.d_state, device=device)

    def forward(self, u: Tensor, edge_index):
        A = -torch.exp(self.A_log)
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )
        x, B, C = torch.split(
            xBC,
            [self.args.d_inner, self.args.d_state, self.args.d_state],
            dim=-1,
        )

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        if self.use_gnn:
            torch.use_deterministic_algorithms(True)
            x1 = rearrange(x, "b l h p -> (b l) (h p)")
            x1 = self.gnn(x1, edge_index)
            torch.use_deterministic_algorithms(False)
            x1 = rearrange(
                x1,
                "(b l) (h p) -> b l h p",
                b=u.shape[0],
                h=self.args.nheads,
                p=self.args.d_state,
            )
            C = C + self.gnn_norm(x1.squeeze(2))
            C = self.projector(C)

        y, _ = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)
        return y, None


def segsum(x: Tensor, device: Device = None) -> Tensor:
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    assert x.shape[1] % chunk_size == 0

    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size)
        for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    return x * torch.sigmoid(x)

"""Exact ATen swept-circle operations with reduced CPU Python dispatch."""
import torch
from torch import Tensor

@torch.jit.script
def swept_tensor(q: Tensor, d: Tensor, radius_squared: Tensor, contact_limit: Tensor, inside: bool, alpha_limit: float) -> Tensor:
    a = d.square().sum(-1)
    b = (q * d).sum(-1)
    c = q.square().sum(-1) - radius_squared
    disc = b.square() - a * c
    root = disc.clamp_min(0).sqrt()
    moving = a > 1.0e-16
    if inside:
        alpha = (-b + root) / a.clamp_min(1.0e-16)
        endpoint = q + d
        valid = moving & (endpoint.norm(p=2.0, dim=[-1]) >= contact_limit) & ((endpoint * d).sum(-1) > 0)
        outside = q.norm(p=2.0, dim=[-1]) >= contact_limit
        alpha = torch.where(outside, 0.0, alpha.clamp_min(0))
        valid = valid | outside
    else:
        alpha = (-b - root) / a.clamp_min(1.0e-16)
        touching = q.norm(p=2.0, dim=[-1]) <= contact_limit
        entering = b < -1.0e-12
        valid = moving & (disc >= 0) & entering & (alpha >= 0)
        alpha = torch.where(touching, 0.0, alpha)
        valid = valid | touching
    return torch.where(valid & (alpha <= alpha_limit), alpha.clamp(0, 1), float('inf'))


@torch.jit.script
def swept_scalar(q: Tensor, d: Tensor, radius_squared: float, contact_limit: float, inside: bool, alpha_limit: float) -> Tensor:
    a = d.square().sum(-1)
    b = (q * d).sum(-1)
    c = q.square().sum(-1) - radius_squared
    disc = b.square() - a * c
    root = disc.clamp_min(0).sqrt()
    moving = a > 1.0e-16
    if inside:
        alpha = (-b + root) / a.clamp_min(1.0e-16)
        endpoint = q + d
        valid = moving & (endpoint.norm(p=2.0, dim=[-1]) >= contact_limit) & ((endpoint * d).sum(-1) > 0)
        outside = q.norm(p=2.0, dim=[-1]) >= contact_limit
        alpha = torch.where(outside, 0.0, alpha.clamp_min(0))
        valid = valid | outside
    else:
        alpha = (-b - root) / a.clamp_min(1.0e-16)
        touching = q.norm(p=2.0, dim=[-1]) <= contact_limit
        entering = b < -1.0e-12
        valid = moving & (disc >= 0) & entering & (alpha >= 0)
        alpha = torch.where(touching, 0.0, alpha)
        valid = valid | touching
    return torch.where(valid & (alpha <= alpha_limit), alpha.clamp(0, 1), float('inf'))


@torch.jit.script
def dense_first(p: Tensor, d: Tensor, centers: Tensor, enabled: Tensor, live: Tensor,
                radii_squared: Tensor, contact_limit: Tensor,
                boundary_squared: Tensor, boundary_limit: Tensor,
                boundary_enabled: Tensor, upper: Tensor, end: int,
                pair_squared: float, pair_limit: float, alpha_limit: float):
    """Original three dense matrices, masks and reductions in one CPU dispatch."""
    q = p[:, :, None] - centers[:, None]
    ct = swept_tensor(q, d[:, :, None], radii_squared, contact_limit, False, alpha_limit)
    ct = torch.where(enabled[:, None] & live[:, :, None], ct, float('inf'))
    boundary_q = (p[:, :, None] - p[:, end:end+1, None]).expand(-1, -1, 2, -1)
    boundary_delta = (d[:, :, None] - d[:, end:end+1, None]).expand(-1, -1, 2, -1)
    bt = swept_tensor(boundary_q, boundary_delta, boundary_squared, boundary_limit, True, alpha_limit)
    pt = swept_scalar(p[:, :, None] - p[:, None], d[:, :, None] - d[:, None], pair_squared, pair_limit, False, alpha_limit)
    bt = torch.where(boundary_enabled & live[:, :, None], bt, float('inf'))
    st = torch.cat((ct, bt), -1)
    pt = torch.where(live[:, :, None] & live[:, None] & upper, pt, float('inf'))
    static_first = st.flatten(1).amin(1)
    first = torch.minimum(static_first, pt.flatten(1).amin(1))
    return st, pt, first

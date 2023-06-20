from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, device = 0, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        #context_dim = default(context_dim, query_dim)
        context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, device = device)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, device = device)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, device = device)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, device = device),
            nn.Dropout(dropout)
        )


    def forward(self, x, context=None, mask=None):
        h = self.heads
        #x = x.unsqueeze(1) FLATTEN
        #context = context.unsqueeze(1)
        q = self.to_q(x)
        #context = default(context, x)
        context = context
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        '''
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        '''
        
        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

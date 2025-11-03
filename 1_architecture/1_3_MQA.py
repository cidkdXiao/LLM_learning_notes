import torch
from torch import nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention:
      - Q: per-head (d_model -> d_model)
      - K,V: shared across heads (d_model -> head_dim)
    输入输出默认形状: [B, L, d_model]
    mask 支持: [B, Lk] / [B, 1, 1, Lk] / [B, 1, Lq, Lk]
    """
    def __init__(self, d_model, num_heads, dropout=0.1, bias=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性投影
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, self.head_dim, bias=bias)  # 共享 K
        self.w_v = nn.Linear(d_model, self.head_dim, bias=bias)  # 共享 V

        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def _prepare_mask(self, mask, B, Lq, Lk, device, dtype):
        """
        将用户传入的各种 mask 统一成可广播到 [B, 1, Lq, Lk] 的布尔屏蔽（True 表示可见，False 表示屏蔽）。
        允许:
          - [B, Lk]        (padding mask)
          - [B, 1, 1, Lk]  (典型 decoder kv mask)
          - [B, 1, Lq, Lk] (已对齐形状)
        返回:
          - mask_bool: [B, 1, Lq, Lk], torch.bool, True=keep, False=mask
        """
        if mask is None:
            return None

        if mask.dtype != torch.bool:
            # 约定: 非 0 即 True
            mask = mask != 0

        if mask.dim() == 2:                 # [B, Lk]
            mask = mask[:, None, None, :]   # -> [B, 1, 1, Lk]
            mask = mask.expand(B, 1, Lq, Lk)
        elif mask.shape == (B, 1, 1, Lk):   # [B,1,1,Lk]
            mask = mask.expand(B, 1, Lq, Lk)
        elif mask.shape == (B, 1, Lq, Lk):  # 已经对齐
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}; expected [B,Lk] or [B,1,1,Lk] or [B,1,Lq,Lk]")

        return mask.to(device=device, dtype=torch.bool)

    def forward(self, query, key, value, mask=None):
        """
        query/key/value: [B, L, d_model]
        返回:
          output: [B, Lq, d_model]
          attn_probs: [B, h, Lq, Lk]
        """
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape
        device = query.device

        # 1) 线性映射
        Q = self.w_q(query)         # [B, Lq, d_model]
        K = self.w_k(key)           # [B, Lk, head_dim]
        V = self.w_v(value)         # [B, Lk, head_dim]

        # 2) 拆头（Q），共享（K,V）
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)   # [B, h, Lq, dh]
        K = K.unsqueeze(1).expand(B, self.num_heads, Lk, self.head_dim)    # [B, h, Lk, dh]
        V = V.unsqueeze(1).expand(B, self.num_heads, Lk, self.head_dim)    # [B, h, Lk, dh]

        # 3) Scaled dot-product (在 float32 中做更稳)
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q.to(torch.float32), K.transpose(-2, -1).to(torch.float32)) * scale  # [B,h,Lq,Lk]

        # 4) mask（True=keep，False=mask）
        mask_bool = self._prepare_mask(mask, B, Lq, Lk, device, attn_scores.dtype)
        if mask_bool is not None:
            # 将被 mask 的位置设为极小值（-inf）
            attn_scores = attn_scores.masked_fill(~mask_bool, torch.finfo(attn_scores.dtype).min)

        # 5) softmax + dropout
        attn_probs = F.softmax(attn_scores, dim=-1).to(V.dtype)  # 回到原 dtype
        attn_probs = self.dropout(attn_probs)

        # 6) 加权求和
        attn_output = torch.matmul(attn_probs.to(torch.float32), V.to(torch.float32))  # [B,h,Lq,dh]
        attn_output = attn_output.to(V.dtype)

        # 7) 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # [B,Lq,d_model]

        # 8) 输出映射
        output = self.w_o(attn_output)  # [B,Lq,d_model]
        return output, attn_probs

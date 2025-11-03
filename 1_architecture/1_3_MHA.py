import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 模块 (支持可选 mask)
    输入与输出形状均为 (B, L, D) —— batch size B，seq_len L，特征维度 D
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 基础参数
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性变换层：用于生成 Q, K, V
        # nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        # PyTorch 默认初始化是:Weight：Kaiming uniform（a=√5），等价于根据 fan_in 做均匀分布; Bias：均匀分布 U(-bound, bound)，其中 bound=1/√fan_in
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出线性层
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: 形状均为 (B, L, D)
        mask: [batch_size, 1, 1, seq_len] 或 None
        """
        B, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()

        # 1️⃣ 线性映射
        Q = self.w_q(query)  # [B, seq_len_q, d_model]
        K = self.w_k(key)  # [B, seq_len_k, d_model]
        V = self.w_v(value)  # [B, seq_len_k, d_model]

        # 2️⃣ 拆分成多个头并转置维度，方便做 batch 矩阵乘法
        # view      [B, seq_len, d_model] -> [B, seq_len, num_heads, head_dim]      拆分注意力头
        # transpose [B, seq_len, num_heads, head_dim] -> [B, num_heads, seq_len, head_dim]  交换维度
        # 使得每个 batch、每个头都有自己的 Q, K, V 矩阵；
        # 并且便于后续计算注意力分数
        Q = Q.view(B, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 3️⃣ 计算 scaled dot-product attention
        # K.transpose(-2, -1)：把 K 的最后两个维度调换，使得形状 [B, num_heads, d_h, L_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # 形状: [B, num_heads, seq_len_q, seq_len_k]

        if mask is not None:
            # mask中为0的位置设为非常小的负值，防止被softmax选中
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # softmax 归一化
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len_q, head_dim]

        # 4️⃣ 拼接所有头的输出
        # transpose back: [B, seq_len_q, num_heads, head_dim]
        # view back:      [B, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous() \
            .view(B, seq_len_q, self.d_model)

        # 5️⃣ 最终线性映射
        output = self.w_o(attn_output)

        return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(0)

    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)  # batch=2, seq_len=10

    out, attn = mha(x, x, x)
    print("输出形状:", out.shape)
    print("注意力权重形状:", attn.shape)
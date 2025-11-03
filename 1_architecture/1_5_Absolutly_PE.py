import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        d_model: 词向量维度（embedding dimension）
        max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 创建 [max_len, d_model] 大小的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位置用 sin，奇数位置用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度方便与输入相加
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不作为可训练参数

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 将前 seq_len 个位置编码加到输入 embedding 上
        return x + self.pe[:, :seq_len, :]

# ====== 使用示例 ======
if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 512
    dummy_input = torch.zeros(batch_size, seq_len, d_model)
    pe = PositionalEncoding(d_model)
    out = pe(dummy_input)
    print(out)  # torch.Size([2, 10, 512])

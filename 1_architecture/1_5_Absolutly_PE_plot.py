import torch
import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

def positional_encoding(max_len=100, d_model=64):
    """
    生成标准的正弦/余弦位置编码矩阵。
    返回形状: [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ====== 参数设置 ======
max_len = 100     # 序列长度（token数量）
d_model = 512      # 向量维度

# 生成位置编码矩阵
pe = positional_encoding(max_len, d_model)
print(pe)
# # ====== 绘图部分 ======
# plt.figure(figsize=(50, 30))
# plt.imshow(pe, aspect='auto', cmap='RdBu_r')  # 红蓝反色热力图（高值红，低值蓝）
# plt.colorbar(label='Encoding value')
# plt.xlabel('Embedding Dimension Index')
# plt.ylabel('Token (Position) Index')
# plt.title(f'Positional Encoding Heatmap (max_len={max_len}, d_model={d_model})')
# plt.tight_layout()
# plt.show()
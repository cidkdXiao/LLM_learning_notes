import torch
from torch import nn

class LN(nn.Module):
    # 初始化
    def __init__(self, normalizaed_shape,  # 在哪个维度上做 LN
                 eps = 1e-5,  # 防止分母为 0
                 elementwise_affine = True):  # 是否使用可学习的缩放因子和偏移因子
        super(LN, self).__init__()
        # 需要对哪个维度的特征做 LN，torch.size 查看维度
        self.normalized_shape = normalizaed_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # 构造可训练的缩放因子和偏置
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalizaed_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalizaed_shape))

    def forward(self, x):
        # 需要做 LN 的维度和输入特征图对应维度的 shape 相同
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        # 需要做 LN 的维度索引
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        # 计算特征图对应维度的均值和方差
        mean = x.mean(dims, keepdim = True)
        mean_x2 = (x**2).mean(dims, keepdim = True)
        var = mean_x2 - mean**2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 线性变换
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm

if __name__ == '__main__':
    x = torch.linespace(0, 23, 24, dtype=torch.float32)  # 构造输入层
    x = x.reshape([2, 3, 2 * 2])

    # 实例化
    ln = LN(x.shape[1:])
    # 前向传播
    x = ln.forward(x)

    print(x.shape)
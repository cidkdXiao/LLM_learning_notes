import torch
from torch import nn

class BN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        """
        num_features: 通道数 C（对每个通道独立归一化）
        eps: 数值稳定项（加在方差上）
        momentum: 更新 running_mean/var 的动量
        affine: 是否使用逐通道的可学习 γ/β
        track_running_stats: 是否跟踪 running_mean/var（推理时使用）
        """
        super().__init__()
        self.num_features = int(num_features)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # γ/β：形状均为 (C,)
        if self.affine:
            self.weight = nn.Parameter(torch.empty(self.num_features))
            self.bias   = nn.Parameter(torch.empty(self.num_features))
            self.reset_parameters_affine()

        # running stats：形状均为 (C,)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var",  torch.ones(self.num_features))
            self.register_buffer("num_batches_tracked", torch.zeros(1, dtype=torch.long))

    def reset_parameters_affine(self):
        # 常见初始化：γ=1，β=0
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def _broadcast(self, t, x):
        """
        将形如 (C,) 的 t 变形为 (1, C, 1, 1) 以便与 NCHW 广播
        对于更高维输入（如 NCDHW），会自动扩展为 (1, C, 1, 1, 1, ...)
        """
        assert t.dim() == 1 and t.numel() == x.size(1)
        shape = [1, x.size(1)] + [1] * (x.dim() - 2)
        return t.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: 形状 (N, C, H, W)；也可扩展到 (N, C, *spatial)
        归一化维度：对每个通道，跨 (N, *spatial) 统计均值/方差
        """
        assert x.dim() >= 2, "expected at least 2D input (N, C, ...)"
        assert x.size(1) == self.num_features, \
            f"expected C={self.num_features}, got C={x.size(1)}"

        # 需要归一化（求均值/方差）的维度：batch + 全部空间维，不包含通道维
        # 例如 NCHW -> dims = (0, 2, 3)
        reduce_dims = (0,) + tuple(range(2, x.dim()))

        if self.training or not self.track_running_stats:
            # —— 训练：用当前 batch 统计量 —— #
            mean = x.mean(dim=reduce_dims, keepdim=True)                         # (1, C, 1, 1, ...)
            var  = x.var(dim=reduce_dims, keepdim=True, unbiased=False)          # (1, C, 1, 1, ...)

            # 更新 running stats（不记录梯度）
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(
                        self.momentum * mean.view(-1)
                    )
                    self.running_var.mul_(1 - self.momentum).add_(
                        self.momentum * var.view(-1)
                    )
                    self.num_batches_tracked += 1
        else:
            # —— 推理：用运行时统计量 —— #
            mean = self._broadcast(self.running_mean, x)                          # (1, C, 1, 1, ...)
            var  = self._broadcast(self.running_var,  x)                          # (1, C, 1, 1, ...)

        # 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # 仿射变换（逐通道）
        if self.affine:
            gamma = self._broadcast(self.weight, x)  # (1, C, 1, 1, ...)
            beta  = self._broadcast(self.bias,   x)  # (1, C, 1, 1, ...)
            y = x_hat * gamma + beta
        else:
            y = x_hat

        return y


if __name__ == "__main__":
    # 构造输入：NCHW = (4, 3, 4, 4)，整数 0~9 后转为 float
    x = torch.randint(0, 10, (4, 3, 4, 4), dtype=torch.int32).float()
    print("输入 x.shape:", tuple(x.shape))  # (4, 3, 4, 4)

    bn = BN2d(num_features=3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    bn.train()  # 训练态：使用当前 batch 统计量

    # 前向
    y = bn(x)
    print("输出 y.shape:", tuple(y.shape))  # (4, 3, 4, 4)

    # —— 验证：按通道跨 (N,H,W) 的均值≈0、方差≈1 —— #
    reduce_dims = (0, 2, 3)         # 对每个通道，跨 N,H,W
    y_mean = y.mean(dim=reduce_dims)                  # 形状: (C,)
    y_var  = y.var(dim=reduce_dims, unbiased=False)   # 形状: (C,)
    print("训练态，每通道均值≈0:", y_mean)
    print("训练态，每通道方差≈1:", y_var)

    # —— 推理态：使用 running stats —— #
    bn.eval()
    y_eval = bn(x)
    print("推理态输出 y_eval.shape:", tuple(y_eval.shape))

    # 与 PyTorch 官方 BN2d 对比（可选校验）
    ref = nn.BatchNorm2d(num_features=3, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    # 对齐参数与 running stats
    with torch.no_grad():
        ref.weight.copy_(bn.weight)
        ref.bias.copy_(bn.bias)
        ref.running_mean.copy_(bn.running_mean)
        ref.running_var.copy_(bn.running_var)

    ref.train()
    y_ref = ref(x)
    print("与官方实现最大绝对误差（训练态）:", (y - y_ref).abs().max().item())

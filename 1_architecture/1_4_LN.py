import torch
from torch import nn

class LN(nn.Module):
    # 初始化
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        normalized_shape: tuple 或 torch.Size，表示在输入的最后若干维上做 LN
            例：(3, 4) -> 对输入 (..., 3, 4) 的最后 2 维做归一化
        eps: 数值稳定项
        elementwise_affine: 是否使用可学习的逐元素 gamma/beta
        """
        super(LN, self).__init__()
        self.normalized_shape = tuple(normalized_shape)  # 统一成 tuple，便于比较
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            """
            1. * 是 参数解包（unpack） 操作符：把一个可迭代对象（如 tuple、list、torch.Size）在调用处拆成一串位置参数传入函数。
                - 若 normalized_shape = (3, 4)，*normalized_shape 会变成 3, 4，因此相当于 torch.Tensor(3, 4) —— 创建一个 形状为 (3, 4) 的（未初始化）张量。
                - 若不加 *，写成 torch.Tensor(normalized_shape)，这会把 元组当作数据本身，得到形状为 (2,)、内容近似 [3., 4.] 的一维张量（明显不对）。
            - 调用处的 *：把可迭代对象“拆开”成多个位置参数。
                ```python
                def f(a, b): ...
                t = (2, 5)
                f(*t)      # 等价于 f(2, 5)
                ```
            - 定义处的 *args：把多余的位置参数“打包”成一个元组。
                ```python
                def f(*args): ...
                f(2, 5)  # 等价于 f((2, 5))
                ```
                
            2. torch.nn.Parameter 是 torch.Tensor 的一个子类，专门用来在 nn.Module 里声明可训练权重。把张量包成 Parameter 并作为模块属性赋值后，它会被自动注册到该模块的参数列表中（model.parameters() / model.named_parameters() 可见），从而被优化器更新。
            - 自动注册：作为 nn.Module 的属性存在时，会出现在 model.parameters() 里。
            - 默认梯度：requires_grad=True（可改）。
            - 参与优化：通常通过 optimizer = torch.optim.SGD(model.parameters(), ...) 被更新。
            - 叶子张量：nn.Parameter 被视为叶子张量（grad_fn=None），不会保留创建它的计算历史。
            - 不想训练：要么设 param.requires_grad = False，要么用 register_buffer 保存为缓冲区（例如 BN 的 running mean/var）。
            - 初始化：不要用 .data 改值；用 with torch.no_grad(): param.copy_(...) 更安全。
            """
            # 形状与 normalized_shape 一致，用于对被归一化的维度做逐元素缩放/平移
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
            self.reset_parameters()

    def reset_parameters(self):
        # 与 PyTorch 内置 LayerNorm 一致: gamma=1, beta=0
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        # x 的末尾维度必须等于 normalized_shape
        assert self.normalized_shape == tuple(x.shape[-len(self.normalized_shape):]), \
            f"expected trailing shape {self.normalized_shape}, but got {tuple(x.shape)}"

        # 需要归一化的维度索引，例如 normalized_shape 有 2 维 -> dims=[-1,-2]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        # 计算均值与方差（按被归一化的维度），保留维度便于广播
        mean = x.mean(dims, keepdim=True)  # 形状: 与 x 相同，但被归一化的维度都缩成 1
        mean_x2 = (x * x).mean(dims, keepdim=True)
        var = mean_x2 - mean * mean  # E(x^2)-E(x)^2, 形状同 mean

        # 标准化（与 x 同形状）
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 线性变换
        if self.elementwise_affine:
            # weight/bias 形状是 normalized_shape，会通过广播作用到 x_norm 的最后若干维
            x_norm = x_norm * self.weight + self.bias

        return x_norm

if __name__ == "__main__":
    """
    torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    在区间 [start, end] 上返回均匀间隔的数字 (函数的作用是，返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量。)
    例：torch.linspace(0, 23, 24) -> tensor([ 0.,  1.,  2., ..., 23.])
    """
    # 构造输入：等距 24 个数，reshape 为 (2,3,4)
    x = torch.linspace(0, 23, 24, dtype=torch.float32).reshape(2, 3, 4)
    print("输入 x.shape:", tuple(x.shape))          # (2, 3, 4)
    print("输入 x:", x)

    # 对 x 的最后两维 (3,4) 做 LayerNorm（即对 C 和 K 一起归一化）
    ln = LN(x.shape[1:])                           # normalized_shape = (3, 4)

    # # 前向
    # y = ln(x)
    # print("输出 y.shape:", tuple(y.shape))         # (2, 3, 4)
    #
    # # —— 验证归一化：按 (3,4) 两维做均值/方差 —— #
    # # 计算每个样本的均值/方差（保留 batch 维度）
    # dims = [-1, -2]  # 对 (3,4) 这两维
    # y_mean = y.mean(dim=dims)                      # 形状: (2,)
    # y_var = y.var(dim=dims, unbiased=False)       # 形状: (2,)
    # print("每个样本归一化后的均值≈0:", y_mean)
    # print("每个样本归一化后的方差≈1:", y_var)
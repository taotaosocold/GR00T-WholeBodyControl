import torch.nn as nn
import torch
from motionbricks.vqvae.neural_modules.resnet import Resnet1D
# encoder部分是纯卷积的时序压缩网络,把运动序列(B,F,T)其中F是输入特征维度逐步下采样并提升通道数,最终得到紧凑的潜在表示,T是帧数
class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width,  # 251
                 output_emb_width = 512,
                 down_t = 3,    # 下采样次数
                 stride_t = 2,  # 卷积步长,这里说明每次移动两帧
                 width = 512,   # 中间层通道数
                 depth = 3,     # 每个下采样的残差块层数
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        # filter_t就是核的大小,pad_t就是填充,卷积核每次覆盖四帧
        filter_t, pad_t = stride_t * 2, stride_t // 2  # stride = 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        input_dim = width
        # 重复down_t次
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                # 时间下采样卷积,每个卷积核大小为[input_dim, filter_t],总共有width个卷积核
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                # Resnet1D是一个时序残差网络模块，在编码器中紧跟在每个下采样卷积之后。它的作用是在不改变时间长度和通道数的前提下，充分提取局部和长程时序特征
                # 输入和输出形状完全一致
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


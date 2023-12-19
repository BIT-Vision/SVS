import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven.surrogate import ATan


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1, bias=False)
        self.fc1_bn = nn.BatchNorm1d(hidden_features, track_running_stats=False)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

        self.fc2_linear = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1, bias=False)
        self.fc2_bn = nn.BatchNorm1d(out_features, track_running_stats=False)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N = x.shape
        x = self.fc1_linear(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N)
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, N)
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(dim / num_heads)
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim, track_running_stats=False)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim, track_running_stats=False)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim, track_running_stats=False)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())
        self.attn_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, v_threshold=0.5, surrogate_function=ATan())

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim, track_running_stats=False)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

    def forward(self, x, kv):
        T, B, C, N = x.shape
        x_for_q = x.flatten(0, 1)
        x_for_kv = kv.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_q)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        k_conv_out = self.k_conv(x_for_kv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        v_conv_out = self.v_conv(x_for_kv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        kv = (k.transpose(-2, -1) @ v) * self.scale
        x = q @ kv

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)

        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, B, C, N))
        return x, None


class Down_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.norm_layer = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.lif = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.embedding(x.flatten(0, 1))
        x = self.norm_layer(x).reshape(T, B, -1, H, W)
        x = self.lif(x).flatten(0, 1)
        x = self.maxpool(x).reshape(T, B, -1, H // 2, W // 2)

        return x


class OR_Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x, kv):
        # T, B, C, HW = x.shape
        x_attn, attn = self.attn(x, kv)
        x = (x + x_attn) - x * x_attn
        x_mlp = self.mlp(x)
        x = (x + x_mlp) - x * x_mlp
        return x, x


class RefineBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=False, surrogate_function=ATan())

    def forward(self, x1, x2):
        T, B, _, H, W = x1.shape
        x1 = self.conv1(x1.flatten(0, 1))
        x1 = self.bn1(x1).reshape(T, B, -1, H, W)
        x1 = self.lif1(x1)
        x2 = F.interpolate(x2.flatten(0, 1), scale_factor=2, mode="nearest")
        x2 = x2.reshape(T, B, -1, H, W)
        out = torch.cat([x1, x2], dim=2)

        out = self.conv2(out.flatten(0, 1))
        out = self.bn2(out).reshape(T, B, -1, H, W)
        out = self.lif2(out)
        return out


class Refine(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.block1 = RefineBlock(dim * 4, dim * 8)
        self.block2 = RefineBlock(dim * 2, dim * 8)

    def forward(self, x1, x2, x3):
        m2 = self.block1(x2, x3)
        m1 = self.block2(x1, m2)
        return m1


class RST(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, step=5, tau=2.0, clip=False):
        super().__init__()
        self.step = step
        self.clip = clip

        hidden_channel = 64

        self.sps1 = Down_Block(in_ch, hidden_channel)
        self.sps2 = Down_Block(hidden_channel, hidden_channel * 2)
        self.sps3 = Down_Block(hidden_channel * 2, hidden_channel * 4)
        self.sps4 = Down_Block(hidden_channel * 4, hidden_channel * 8)

        self.block = nn.ModuleList([OR_Block(dim=hidden_channel * 8, num_heads=8, mlp_ratio=4) for i in range(6)])

        self.refine = Refine(hidden_channel)

        self.conv2d = nn.Conv2d(hidden_channel * 8, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward_one(self, x, last_mem=None):
        x = (x.unsqueeze(0)).repeat(self.step, 1, 1, 1, 1)
        _, b, _, h, w = x.shape

        x1 = self.sps1(x)
        x2 = self.sps2(x1)  # T,b,c2,h//4,w//4
        x3 = self.sps3(x2)
        x4 = self.sps4(x3)

        current_mem = x4.flatten(3)
        bottle = x4.flatten(3)
        kv = bottle

        for i, block in enumerate(self.block):
            kv = torch.cat([kv[1:], kv[-1:]], dim=0)
            bottle, kv = block(bottle, kv)

        bottle = bottle.reshape(self.step, b, -1, h // 16, w // 16)

        result = self.refine(x2, x3, bottle)

        if self.clip:
            out_list = list()
            for item in result:
                out = self.conv2d(item)
                out = F.interpolate(out, size=(h, w), mode="bilinear")
                out_list.append(out)
            out = torch.cat(out_list, dim=1)
        else:
            out = self.conv2d(result.mean(0))
            out = F.interpolate(out, size=(h, w), mode="bilinear")

        return out, current_mem, None, None

    def forward(self, x, last_mem=None):
        if x.dim() == 4:
            out, middle, concat, _ = self.forward_one(x, last_mem)
            return out.unsqueeze(0), None, None, None
        else:
            out_list = list()
            for item in x:
                out, last_mem, concat, _ = self.forward_one(item, last_mem)
                out_list.append(out)
            output = torch.stack(out_list, dim=0)
            return output, last_mem, None, None


def build_RST(args, cfg):
    step = 5 if args.step else 1
    model = RST(in_ch=cfg.MODEL.INPUT_DIM, out_ch=cfg.MODEL.OUTPUT_DIM, step=step, tau=2.0, clip=args.clip)
    return model
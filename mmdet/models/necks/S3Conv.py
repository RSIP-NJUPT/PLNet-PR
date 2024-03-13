from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath

use_sync_bn = True

def conv_bn_ori(
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True
):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
    )

    if bn:
        result.add_module("bn", get_bn(out_channels))
    return result

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    if isinstance(kernel_size, int) or len(set(kernel_size)) == 1:
        return conv_bn_ori(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation,
            bn)
    else:
        return ConvGroupShift(in_channels, out_channels, kernel_size, stride, groups, bn)
    
def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def get_conv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
):
    # return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    try:
        paddings = (kernel_size[0] // 2, kernel_size[1] // 2)
    except:
        paddings = padding
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, paddings, dilation, groups, bias
    )


def get_bn(channels):
    return nn.BatchNorm2d(channels)

class ConvGroupShift(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel, stride=1, group=1,
                 bn=True, weight=None):
        super().__init__()
        assert len(set(kernel)) == 2, "must have two different kernel size"
        mink, maxk = min(kernel), max(kernel)
        self.kernels = kernel
        self.stride = stride
        if (mink, maxk) == kernel:
            self.VH = 'H'  # 横向
        else:
            self.VH = 'V'
        padding, after_padding_index, index = self.shift(kernel)
        padding = (padding, mink // 2) if self.VH == 'V' else (mink // 2, padding)
        self.pad = after_padding_index, index
        print(padding, after_padding_index, index)
        self.nk = math.ceil(maxk / mink)
        self.split_convs = nn.Conv2d(in_channels, out_channels * self.nk,
                                     kernel_size=mink,  stride=stride,
                                     padding=padding, groups=group,
                                     bias=False)
        self.reloc_weight(weight)
        self.use_bn = bn
        if bn:
            self.bn = get_bn(out_channels)
            

    def shift(self, kernels):
        '''
        Regardless of the step size, the convolution can slide the window to the boundary at most
        '''
        mink, maxk = min(kernels), max(kernels)
        mid_p = maxk // 2
        # 1. new window size is mink. middle point index in the window
        offset_idx_left = mid_p % mink
        offset_idx_right = (math.ceil(maxk / mink) * mink - mid_p - 1) % mink
        # 2. padding
        padding = offset_idx_left % mink
        while padding < offset_idx_right:
            padding += mink
        # 3. make sure last pixel can be scan by min window
        while padding < (mink - 1):
            padding += mink
        # 4. index of windows start point of middle point
        after_padding_index = padding - offset_idx_left
        index = math.ceil((mid_p + 1) / mink)
        real_start_idx = index - after_padding_index // mink
        return padding, after_padding_index, real_start_idx

        # todo:stride 时候的extra_padding

    def forward(self, inputs):
        out = self.split_convs(inputs)
        b, c, h, w = out.shape
        # split output
        *_, ori_h, ori_w = inputs.shape
        # out = torch.split(out, c // self.nk, 1)
        out = torch.split(out.reshape(b, -1, self.nk, h, w), 1, 2)  # ※※※※※※※※※※※
        x = 0
        for i in range(self.nk):
            outi = self.rearrange_data(out[i], i, ori_h, ori_w)
            x = x + outi
        if self.use_bn:
            x = self.bn(x)
        return x

    def rearrange_data(self, x, idx, ori_h, ori_w):
        pad, index = self.pad
        x = x.squeeze(2)  # ※※※※※※※
        *_, h, w = x.shape
        k = min(self.kernels)
        ori_k = max(self.kernels)
        ori_p = ori_k // 2
        stride = self.stride
        # need to calculate start point after conv
        # how many windows shift from real start window index
        if (idx + 1) >= index:
            pad_l = 0
            s = (idx + 1 - index) * (k // stride)
        else:
            pad_l = (index - 1 - idx) * (k // stride)
            s = 0
        if self.VH == 'H':
            # assume add sufficient padding for origin conv
            suppose_len = (ori_w + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (w + pad_l) else s + suppose_len - w - pad_l
            new_pad = (pad_l, pad_r, 0, 0)
            dim = 3
            e = w + pad_l + pad_r - s - suppose_len
        else:
            # assume add sufficient padding for origin conv
            suppose_len = (ori_h + 2 * ori_p - ori_k) // stride + 1
            pad_r = 0 if (s + suppose_len) <= (h + pad_l) else s + suppose_len - h - pad_l
            new_pad = (0, 0, pad_l, pad_r)
            dim = 2
            e = h + pad_l + pad_r - s - suppose_len
        # print('new_pad', new_pad)
        if len(set(new_pad)) > 1:
            x = F.pad(x, new_pad)
        split_list = [s, suppose_len, e]
        # print('split_list', split_list)
        xs = torch.split(x, split_list, dim=dim)
        return xs[1]

    def reloc_weight(self, w):
        if w is None: return
        c1, c2, k1, k2 = self.split_convs.weight.data.shape

        if self.VH == 'H':
            pad_r = k1 - w.shape[3] % k1
            w = F.pad(w, (0, pad_r, 0, 0))
            w = torch.split(w.unsqueeze(1), k1, dim=3 + 1)  # ※※※※※※※※
            w = torch.cat(w, dim=1).reshape(-1, c2, k1, k2)  # ※※※※※※※※
        else:
            pad_r = k1 - w.shape[2] % k1
            w = F.pad(w, (0, 0, 0, pad_r))
            w = torch.split(w.unsqueeze(1), k1, dim=2 + 1)
            w = torch.cat(w, dim=1).reshape(-1, c2, k1, k2)
        self.split_convs.weight.data = w

class ReparamLargeKernelConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            small_kernel,
            small_kernel_merged=False,
            Decom=False,
            bn=True,
    ):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.Decom = Decom
        padding = kernel_size // 2
        if small_kernel_merged:  
            self.lkb_reparam = get_conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            if self.Decom:
                self.LoRA1 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, small_kernel),
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )
                self.LoRA2 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(small_kernel, kernel_size),
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )
            else:
                self.lkb_origin = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=groups,
                    bn=bn,
                )

            if (small_kernel is not None) and small_kernel < kernel_size:
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bn=bn,
                )

    def forward(self, inputs):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        elif self.Decom:
            out = self.LoRA1(inputs) + self.LoRA2(inputs)
            if hasattr(self, "small_conv"):
                out = out+self.small_conv(inputs)
            out = out + inputs
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")




class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        super(SKConv,self).__init__()
        d=max(in_channels//r,L) 
        self.M=M
        self.out_channels=out_channels
        self.global_pool=nn.AdaptiveAvgPool2d(1) 
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),

                               nn.ReLU(inplace=True))  
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  
        self.softmax=nn.Softmax(dim=1) 
    def forward(self, x):
        batch_size=x[0].size(0)
        output=x

        U=reduce(lambda x,y:x+y,output) 
        s=self.global_pool(U)
        z=self.fc1(s)  
        a_b=self.fc2(z) 
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) 
        a_b=self.softmax(a_b) 

        a_b=list(a_b.chunk(self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 

        return a_b

class S3Conv(nn.Module):
    def __init__(self,out_channels=256, kernels=[27, 21, 15, 9],drop_path=0.5) -> None:
        super().__init__()
        self.kernels = kernels
        dim = out_channels
        self.S2Conv = nn.ModuleList([ReparamLargeKernelConv(
            in_channels=dim,
            out_channels=dim,
            kernel_size=k,
            stride=1,
            groups=dim,
            small_kernel=3,
            small_kernel_merged=False,
            Decom=True,
            bn=True,
        ) for k in kernels])
        self.attn_conv = nn.ModuleList([nn.Conv2d(out_channels, out_channels//4, 1) for _ in range(len(kernels))])
        self.conv_squeeze = nn.Conv2d(2, len(kernels), 7, padding=3)
        self.conv = nn.Conv2d(dim//len(kernels), dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.channel_conv = SKConv(in_channels=out_channels//len(kernels),
                                    out_channels=out_channels//len(kernels),
                                    stride=1,M=len(kernels),r=1,L=32)    

    def forward(self, x):
        res = x
        attn = [block(x) for block in self.S2Conv]            # [dim]
        attn_spatial = [attn_conv(attn[i]) for i, attn_conv in enumerate(self.attn_conv)]      # [dim//4]
        attn_channel = self.channel_conv(attn_spatial)

        attn_cat = torch.cat(attn_spatial, dim=1)
        avg_attn = torch.mean(attn_cat, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn_cat, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn_spatial = [attn_spatial[i] * sig[:,i,:,:].unsqueeze(1) for i in range(len(self.kernels))]  # [dim//4]
        attn = list(map(lambda x, y : x*y, attn_spatial, attn_channel))
        attn = reduce(lambda x,y:x+y, attn)
        attn = self.conv(attn)
        x = res + self.drop_path(x * attn)
        return x
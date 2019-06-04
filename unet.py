import torchvision
import torch
import torch.nn as nn
from fastai.callbacks import *
from torchvision.models import vgg11_bn

###########################
# VGG LOSS
# #############################


def requires_grad(m, b=None):
    "If `b` is not set return `requires_grad` of first param, else set `requires_grad` on all params as `b`"
    ps = list(m.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad=b

class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.m_feat = vgg11_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)

        blocks = [i - 1 for i, o in enumerate(children(self.m_feat)) if isinstance(o, nn.MaxPool2d)]
        layer_ids = blocks[2:5]

        self.wgts = [5, 15, 2]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids)) ] \
                            + [f'gram_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return (x @ x.transpose(1, 2)) / (c * h * w)

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        # L1 loss (mean-absolute error)
        self.feat_losses = [self.base_loss(input, target)]

        # VGG perceptual loss
        self.feat_losses += [self.base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        # Style loss
        self.feat_losses += [self.base_loss(self.gram_matrix(f_in), self.gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))

        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()



class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

###########################
# BRIDGE
# #############################

class Bridge_640(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    torch.Size([5, 512, 2, 2])
    """

    def __init__(self, num_hid, v_dim):
        super().__init__()
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.bridge = nn.Sequential(
            nn.ConvTranspose2d(num_hid, num_hid, kernel_size=2, stride=2),
            ConvBlock(num_hid, v_dim, kernel_size=3),
            ConvBlock(v_dim, v_dim, kernel_size=3),
        )

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1, 1, 1)
        return self.bridge(x)

class Bridge_641(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    torch.Size([5, 512, 2, 2])
    """

    def __init__(self, num_hid, v_dim):
        super().__init__()
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.bridge = nn.Sequential(
            ConvBlock(num_hid, v_dim, kernel_size=2),
            ConvBlock(v_dim, v_dim, kernel_size=3),
        )

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1, 1, 1)
        return self.bridge(x)


class Bridge_1280(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    torch.Size([5, 512, 2, 2])
    """

    def __init__(self, num_hid, v_dim):
        super().__init__()
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.bridge = nn.Sequential(
            nn.ConvTranspose2d(num_hid, num_hid, kernel_size=2, stride=2),
            ConvBlock(num_hid, v_dim, kernel_size=3),
            nn.ConvTranspose2d(v_dim, v_dim, kernel_size=2, stride=2),
            ConvBlock(v_dim, v_dim, kernel_size=3),
        )

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1, 1, 1)
        return self.bridge(x)

class Bridge_1281(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    torch.Size([5, 512, 2, 2])
    """

    def __init__(self, num_hid, v_dim):
        super().__init__()
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.bridge = nn.Sequential(
            ConvBlock(num_hid, v_dim, kernel_size=2),
            nn.ConvTranspose2d(v_dim, v_dim, kernel_size=2, stride=2),
            ConvBlock(v_dim, v_dim, kernel_size=3),
        )

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1, 1, 1)
        return self.bridge(x)

class Bridge_128(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    torch.Size([5, 512, 2, 2])
    """

    def __init__(self, num_hid, v_dim):
        super().__init__()
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.bridge = nn.Sequential(
            nn.ConvTranspose2d(num_hid, v_dim, kernel_size=2, stride=2),
            ConvBlock(v_dim, v_dim, kernel_size=3),

        )

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1, 1, 1)
        return self.bridge(x)


##############################################################
# UNET
##############################################################

class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x



class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 7

    def __init__(self, finetune=False, dropout_unet=0.0, n_classes=3):
        super().__init__()
        self.finetune=finetune

        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        if not self.finetune:
            self.resnet = self.resnet.eval()
            requires_grad(self.resnet, False)

        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(self.resnet.children()))[:3]
        self.input_pool = list(self.resnet.children())[3]
        for bottleneck in list(self.resnet.children())[4:-2]: #dont take last pooling and FC
            # if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet50(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 32, out_channels=32,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=16 + 3, out_channels=16,
                                                    up_conv_in_channels=32, up_conv_out_channels=16))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(16, n_classes, kernel_size=1, stride=1)

        self.dropout_unet = nn.Dropout(p=dropout_unet)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        output_feature_map = x
        x = self.out(x)
        # v = []
        # import numpy as np
        # import pickle
        # for r in pre_pools.keys():
        #     print(r, pre_pools[r].cpu().detach().numpy().astype(np.float16).shape)

        # pickle.dump(v,open("ok3",'wb+'))
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

    def encode(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        return x, pre_pools

    def decode(self,x,pre_pools):
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key].cuda())
            x = self.dropout_unet(x)
        x = self.out(x)
        return x

# #
# model = UNetWithResnet50Encoder().cuda()
# inp = torch.rand((100, 3, 64, 64)).cuda()
# out = model.decode(inp)
# print("####")
# print(out.shape)
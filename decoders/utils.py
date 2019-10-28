import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = True, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.use_batchnorm = use_batchnorm

        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        if (self.use_batchnorm):
            out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        if (self.use_batchnorm):
            out = self.bn2(out)
        out = self.act_func(out)
        return out

###########################################################################
############################ MAIN MODEL BLOCKS ############################
###########################################################################

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)       

###########################################################################
########################## PYRAMID POOLING BLOCKS #########################
###########################################################################

class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_list, in_channels, size=(128, 128), mode='bilinear'):
        super(PyramidPoolingModule, self).__init__()
        self.pool_list = pool_list
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0) for _ in range(len(pool_list))])
        if mode == 'bilinear':
            self.upsample = nn.Upsample(size=size, mode=mode, align_corners=True)
        else:
            self.upsample = nn.Upsample(size=size, mode=mode)
            # self.conv2 = nn.Conv2d(in_channels * (1 + len(pool_list)), in_channels, kernel_size=1)

    def forward(self, x):
        cat = [x]
        for (k, s), conv in zip(self.pool_list, self.conv1):
            out = F.avg_pool2d(x, kernel_size=k, stride=s)
            out = conv(out)
            out = self.upsample(out)
            cat.append(out)
        out = torch.cat(cat, 1)
        # out = self.conv2(out)
        # out = self.relu(out)
        return out


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, depth, atrous_rates=[12, 24, 36]):
        super(AtrousSpatialPyramidPooling, self).__init__()
        self.conv1 = ConvBn2d(in_channels, depth, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = ConvBn2d(in_channels, depth, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0])
        self.conv3_2 = ConvBn2d(in_channels, depth, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1])
        self.conv3_3 = ConvBn2d(in_channels, depth, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.im_conv1 = ConvBn2d(in_channels, depth, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.final_conv1 = ConvBn2d(depth * 5, depth, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        rate1 = self.conv1(x)
        rate2 = self.conv3_1(x)
        rate3 = self.conv3_2(x)
        rate4 = self.conv3_3(x)

        im_level = self.pool(x)
        im_level = self.im_conv1(im_level)
        im_level = self.upsample(im_level)

        out = torch.cat([
            rate1,
            rate2,
            rate3,
            rate4,
            im_level
        ], 1)

        out = self.final_conv1(out)
        return out


###########################################################################
############################ EMANet BLOCKS #################################
###########################################################################

class EMAModule(nn.Module):

    def __init__(self, channels, K, lbda=1, alpha=0.1, T=3):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.lbda = lbda
        self.conv1_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv1_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.bn_out = nn.BatchNorm2d(channels)
        self.register_buffer('bases', torch.empty(K, channels)) # K x C
        # self.bases = Parameter(torch.empty(K, channels), requires_grad=False) # K x C
        nn.init.kaiming_uniform_(self.bases, a=math.sqrt(5))
        # self.bases.data = F.normalize(self.bases.data, dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.conv1_in(x).view(B, C, -1).transpose(1, -1) # B x N x C

        bases = self.bases[None, ...]
        x_in = x.detach()
        for i in range(self.T):
            # Expectation
            if i == (self.T - 1):
                x_in = x
            z = torch.softmax(self.lbda * torch.matmul(x_in, bases.transpose(1, -1)), dim=-1)  # B x N x K
            # Maximization
            bases = torch.matmul(z.transpose(1, 2), x_in) / (z.sum(1)[..., None] + 1e-12) # B x K x C
            bases = F.normalize(bases, dim=-1)
        if self.training:
            self.bases.data = (1 - self.alpha) * self.bases + self.alpha * bases.detach().mean(0)

        x = torch.matmul(z, bases).transpose(1, -1).view(B, C, H, W)
        x = self.conv1_out(x)
        x = self.bn_out(x)
        x += residual
        return x

class GlobalAttentionUpsample(nn.Module):
    def __init__(self, skip_channels, channels, out_channels=None):
        super(GlobalAttentionUpsample, self).__init__()
        self.out_channels = out_channels
        if out_channels is None:
            out_channels = channels
        self.conv3 = nn.Conv2d(skip_channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(channels, channels, kernel_size=1, padding=0)
        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if out_channels is not None:
            self.conv_out = ConvBn2d(channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, skip, up=True):
        # Reduce channels
        skip = self.conv3(skip)
        # Upsample
        if up:
            x = self.upsample(x)
        # GlobalPool and conv1
        cal1 = self.GPool(x)
        cal1 = self.conv1(cal1)
        cal1 = self.relu(cal1)

        # Calibrate skip connection
        skip = cal1 * skip
        # Add
        x = x + skip
        if self.out_channels is not None:
            x = self.conv_out(x)
        return x


###########################################################################
############################ PANet BLOCKS #################################
###########################################################################

class FeaturePyramidAttention(nn.Module):
    def __init__(self, channels, out_channels=None):
        super(FeaturePyramidAttention, self).__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1p = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv3a = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv5b = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.conv7a = nn.Conv2d(channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv7b = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mode='std'):
        H, W = x.shape[2:]
        # down-path
        if mode == 'std':
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == 'reduced':
            xup1 = self.conv7a(x)
        elif mode == 'extended':
            xup1 = F.avg_pool2d(x, kernel_size=4, stride=4)
            xup1 = self.conv7a(xup1)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        
        xup2 = self.upsample(xup3).view(xup2.shape) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode='bilinear', align_corners=True)

        # Merge
        if mode == 'std':
            x1 = self.upsample(xup1) * x1
        elif mode == 'reduced':
            x1 = xup1 * x1
        elif mode == 'extended':
            x1 = F.upsample(xup1, scale_factor=4, mode='bilinear', align_corners=True) * x1
        x1 = x1 + gp
        return x1


class FeaturePyramidAttention_v2(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        self.conv1 = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)
        self.conv1p = ConvBn2d(channels, out_channels, kernel_size=1, stride=1, padding=0, act=True)

        self.conv3a = ConvBn2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True)
        self.conv3b = ConvBn2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True)

        self.conv5a = ConvBn2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True)
        self.conv5b = ConvBn2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, act=True)

        self.conv7a = ConvBn2d(channels, out_channels, kernel_size=7, stride=1, padding=3, act=True)
        self.conv7b = ConvBn2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, act=True)

        self.GPool = nn.AdaptiveAvgPool2d(output_size=1)

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, mode='std'):
        H, W = x.shape[2:]
        # down-path
        if mode == 'std':
            xup1 = self.downsample(x)
            xup1 = self.conv7a(xup1)
        elif mode == 'reduced':
            xup1 = self.conv7a(x)

        xup2 = self.downsample(xup1)
        xup2 = self.conv5a(xup2)

        xup3 = self.downsample(xup2)
        xup3 = self.conv3a(xup3)

        # Skips
        x1 = self.conv1(x)
        xup1 = self.conv7b(xup1)
        xup2 = self.conv5b(xup2)
        xup3 = self.conv3b(xup3)

        # up-path
        xup2 = self.upsample(xup3) + xup2
        xup1 = self.upsample(xup2) + xup1

        # Global Avg Pooling
        gp = self.GPool(x)
        gp = self.conv1p(gp)
        gp = F.upsample(gp, size=(H, W), mode='bilinear', align_corners=True)

        # Merge
        if mode == 'std':
            x1 = self.upsample(xup1) * x1
        elif mode == 'reduced':
            x1 = xup1 * x1
        x1 = x1 + gp
        return x1
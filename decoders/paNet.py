from .utils import *

#TODO refactor to left a generic encoder
class PANetResNet34(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 256
        self.encoder2 = self.encoder.layer2  # 512
        self.encoder3 = self.encoder.layer3  # 1024
        self.encoder4 = self.encoder.layer4  # 2048

        self.center_conv = nn.Sequential(
            L.ConvBn2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(64)

        self.decoder5 = L.GlobalAttentionUpsample(256, 64)
        self.decoder4 = L.GlobalAttentionUpsample(128, 64)
        self.decoder3 = L.GlobalAttentionUpsample(64, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=3, stride=2, padding=1)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        e4 = self.center_conv(e4)
        f = self.FPA(e4, mode='std')

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        logit = self.upsample(self.logit(d3))

        return logit

    def loss(self, criterion, logit, mask):
        return criterion(logit, mask)


class PANetResNet50(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = resnet50(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.center_conv = nn.Sequential(
            L.ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention_v2(512, 64)

        self.decoder5 = L.GlobalAttentionUpsample(1024, 64)
        self.decoder4 = L.GlobalAttentionUpsample(512, 64)
        self.decoder3 = L.GlobalAttentionUpsample(256, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/32

        # if H < 512:
        #     mode = 'reduced'
        # else:
        #     mode = 'std'

        f = self.center_conv(e4)  # 1/32
        f = self.FPA(f, mode='std')

        d5 = self.decoder5(f, e3, up=True)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        # s = F.dropout2d(d, p=0.40, training=self.is_training)

        logit = self.upsample(self.logit(d3))

        return logit

    def loss(self, criterion, logit, mask):
        return criterion(logit, mask)


class PANetDilatedResNet34(nn.Module):

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.encoder = dilated_resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.encoder1 = self.encoder.layer1  # 64
        self.encoder2 = self.encoder.layer2  # 128
        self.encoder3 = self.encoder.layer3  # 256
        self.encoder4 = self.encoder.layer4  # 512

        self.center_conv = nn.Sequential(
            L.ConvBn2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.FPA = L.FeaturePyramidAttention(64)

        self.decoder5 = L.GlobalAttentionUpsample(256, 64)
        self.decoder4 = L.GlobalAttentionUpsample(128, 64)
        self.decoder3 = L.GlobalAttentionUpsample(64, 64)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        e0 = self.conv1(x)  # 1/2
        p = F.max_pool2d(e0, kernel_size=2, stride=2)  # 1/4

        e1 = self.encoder1(p)  # 1/4
        e2 = self.encoder2(e1)  # 1/8
        e3 = self.encoder3(e2)  # 1/16
        e4 = self.encoder4(e3)  # 1/16

        # if H < 512:
        #     mode = 'reduced'
        # else:
        #     mode = 'std'

        f = self.center_conv(e4)  # 1/16
        f = self.FPA(f, mode='std')

        d5 = self.decoder5(f, e3, up=False)  # 1/16
        d4 = self.decoder4(d5, e2, up=True)  # 1/8
        d3 = self.decoder3(d4, e1, up=True)  # 1/4

        # s = F.dropout2d(d, p=0.40, training=self.is_training)

        logit = self.upsample(self.logit(d3))

        return logit

    def loss(self, criterion, logit, mask):
        return criterion(logit, mask)
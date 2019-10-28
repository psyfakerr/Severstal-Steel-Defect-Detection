from .utils import *


class UnetPlusPlusDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            deepsupervision=False
    ):
        super().__init__()

        self.deepsupervision = deepsupervision
        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = encoder_channels
        out_channels = decoder_channels

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer0_1 = VGGBlock(in_channels[4] + in_channels[3], out_channels[4], use_batchnorm=use_batchnorm)

        self.layer1_1 = VGGBlock(in_channels[3] + in_channels[2], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer0_2 = VGGBlock(in_channels[4] + out_channels[4] + out_channels[3], out_channels[4], use_batchnorm=use_batchnorm)

        self.layer2_1 = VGGBlock(in_channels[2] + in_channels[1], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer1_2 = VGGBlock(in_channels[3] + out_channels[3] + out_channels[2], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer0_3 = VGGBlock(in_channels[4] + out_channels[4] * 2 + out_channels[3], out_channels[4], use_batchnorm=use_batchnorm)

        self.layer4_0 = DecoderBlock(in_channels[0] + in_channels[1], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer3_1 = VGGBlock(in_channels[1] + out_channels[0], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer2_2 = VGGBlock(in_channels[2] + out_channels[2] + out_channels[1], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer1_3 = VGGBlock(in_channels[3] + out_channels[2] + out_channels[3] * 2, out_channels[3], use_batchnorm=use_batchnorm)
        self.layer0_4 = VGGBlock(in_channels[4] + out_channels[3] + out_channels[4] * 3, out_channels[4], use_batchnorm=use_batchnorm)

        if deepsupervision:
            self.final1 = nn.Conv2d(out_channels[4], final_channels, kernel_size=1)
            self.final2 = nn.Conv2d(out_channels[4], final_channels, kernel_size=1)
            self.final3 = nn.Conv2d(out_channels[4], final_channels, kernel_size=1)
            self.final4 = nn.Conv2d(out_channels[4], final_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(out_channels[4], final_channels, kernel_size=1)

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]

        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x0_0 = skips[3]

        x1_0 = skips[2]
        x0_1 = self.layer0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = skips[1]
        x1_1 = self.layer1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.layer0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = skips[0]
        x2_1 = self.layer2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.layer1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.layer0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.layer4_0([encoder_head, skips[0]])
        x3_1 = self.layer3_1(torch.cat([x3_0, x4_0], 1))
        x2_2 = self.layer2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.layer1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.layer0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x1_3.detach()
        x0_4 = self.up(x0_4)

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            
            return output
from encoders import get_encoder
from decoders import *


class SegmentationModel(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            decoder_name='unet',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False  # usefull for VGG models
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = get_decoder (
            decoder_name,
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)

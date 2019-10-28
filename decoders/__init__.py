from .unet import UnetDecoder
from .unetPlusPlus import UnetPlusPlusDecoder
from .utils import *

decoders = {
    'unet': UnetDecoder,
    'unet++': UnetPlusPlusDecoder
}

def get_decoder(
        name,
        encoder_channels,
        decoder_channels,
        final_channels,
        use_batchnorm,
        center):
    decoder = decoders[name]

    return decoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        final_channels=final_channels,
        use_batchnorm=use_batchnorm,
        center=center)
from .ELAMethod import ELAMethod
from .FFTMethod import FFTMethod
from .CLIPZeroShotMethod import CLIPZeroShotMethod
from .CNNStyleGANMethod import CNNStyleGANMethod
from .ConvNeXtStyleGANMethod import ConvNeXtStyleGANMethod

BASE_MODELS = [
    ELAMethod(),
    FFTMethod(),
    CLIPZeroShotMethod(),
    CNNStyleGANMethod(),
    ConvNeXtStyleGANMethod()
]

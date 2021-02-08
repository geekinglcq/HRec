from .deepfm import DeepFM
from .deepmf import DMF
from .fism import FISM
from .ncf import NCF
from .xdeepfm import xDeepFM
from .dssm import DSSM
from .afm import AFM
from .dcn import DCN
from .widedeep import WideDeep
from .nais import NAIS
from .cccf import CCCFNet
from .ddtcdr import DDTCDR
from .autoint import AutoInt
from .duration import DURation

from .utils import ModelType

model_map = {
    # General Model
    "DMF": DMF,
    "FISM": FISM,
    "NCF": NCF,
    # Context Model
    "DeepFM": DeepFM,
    "xDeepFM": xDeepFM,
    "DCN": DCN,
    "AFM": AFM,
    "DSSM": DSSM,
    "WideDeep": WideDeep,
    "NAIS": NAIS,
    "AutoInt": AutoInt,
    # Heterogeneous Model
    "CCCF": CCCFNet,
    "DDTCDR": DDTCDR,
    "DURation": DURation,
}

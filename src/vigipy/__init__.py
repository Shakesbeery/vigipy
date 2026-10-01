import logging

from .PRR import prr
from .ROR import ror
from .RFET import rfet
from .GPS import gps
from .BCPNN import bcpnn
from .LASSO import lasso
from .SCORE import score_da, score_ddi
from .utils import convert, convert_binary, convert_multi_item, convert_ddi
from .LongitudinalModel.LongitudinalModel import LongitudinalModel
from .analyze import analyze, analyze_all, get_default_config
from .consensus import consensus_analysis, ConsensusResult
from .config import (
    PRRConfig, RORConfig, RFETConfig, BCPNNConfig, GPSConfig, LASSOConfig,
    SCOREConfig, SCOREDDIConfig, MethodConfig,
)

logger = logging.getLogger("vigipy")
logger.addHandler(logging.NullHandler())

__version__ = "3.3.1"

__all__ = [
    "__version__",
    "logger",
    # Original function API
    "prr", "ror", "rfet", "gps", "bcpnn", "lasso", "score_da", "score_ddi",
    # Data conversion
    "convert", "convert_binary", "convert_multi_item", "convert_ddi",
    # Longitudinal
    "LongitudinalModel",
    # Unified API
    "analyze", "analyze_all", "get_default_config",
    "PRRConfig", "RORConfig", "RFETConfig",
    "BCPNNConfig", "GPSConfig", "LASSOConfig", "SCOREConfig", "SCOREDDIConfig",
    "MethodConfig",
    # Consensus Analysis
    "consensus_analysis", "ConsensusResult",
]

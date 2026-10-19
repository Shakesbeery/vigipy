import logging

from .PRR import prr
from .ROR import ror
from .RFET import rfet
from .GPS import gps
from .ZIGPS import zigps
from .Sequential import maxsprt, cusum_da
from .BCPNN import bcpnn
from .LASSO import lasso
from .SCORE import score_da, score_ddi
from .AEGIS import aegis
from .utils import convert, convert_binary, convert_multi_item, convert_ddi
from .LongitudinalModel.LongitudinalModel import LongitudinalModel
from .utils.Container import AnalysisResult, DataContainer
from .analyze import analyze, analyze_all, get_default_config
from .consensus import consensus_analysis, ConsensusResult
from .config import (
    BaseConfig,
    PRRConfig, RORConfig, RFETConfig, BCPNNConfig, GPSConfig, ZIGPSConfig,
    MaxSPRTConfig, CUSUMConfig, LASSOConfig, SCOREConfig, SCOREDDIConfig,
    AEGISConfig, MethodConfig,
)

logger = logging.getLogger("vigipy")
logger.addHandler(logging.NullHandler())

__version__ = "3.5.0"

__all__ = [
    "__version__",
    "logger",
    # Original function API
    "prr", "ror", "rfet", "gps", "zigps", "maxsprt", "cusum_da",
    "bcpnn", "lasso", "score_da", "score_ddi", "aegis",
    # Data conversion and containers
    "convert", "convert_binary", "convert_multi_item", "convert_ddi",
    "DataContainer", "AnalysisResult",
    # Longitudinal
    "LongitudinalModel",
    # Unified API
    "analyze", "analyze_all", "get_default_config",
    "BaseConfig",
    "PRRConfig", "RORConfig", "RFETConfig",
    "BCPNNConfig", "GPSConfig", "ZIGPSConfig", "MaxSPRTConfig", "CUSUMConfig",
    "LASSOConfig", "SCOREConfig", "SCOREDDIConfig", "AEGISConfig",
    "MethodConfig",
    # Consensus Analysis
    "consensus_analysis", "ConsensusResult",
]

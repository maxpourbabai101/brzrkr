"""Model subpackage: LSTM, transformer, sentiment, volatility, ensemble, XGBoost."""

from src.model.lstm_model import LSTMPredictor, LSTMConfig
from src.model.transformer_backbone import TransformerPredictor, TransformerConfig
from src.model.sentiment_encoder import SentimentEncoder
from src.model.volatility_module import Garch11, IVSurface
from src.model.ensemble import EnsemblePredictor, EnsembleWeights
from src.model.xgb_predictor import maybe_load_xgb_predictor, XGBoostPredictor
from src.model.momentum_model import MultiFactorMomentum
from src.model.regime_model import RegimeAwareModel

__all__ = [
    "LSTMPredictor",
    "LSTMConfig",
    "TransformerPredictor",
    "TransformerConfig",
    "SentimentEncoder",
    "Garch11",
    "IVSurface",
    "EnsemblePredictor",
    "EnsembleWeights",
    "maybe_load_xgb_predictor",
    "XGBoostPredictor",
    "MultiFactorMomentum",
    "RegimeAwareModel",
]
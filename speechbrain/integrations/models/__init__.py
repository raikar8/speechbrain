"""
Package with models from stand-alone repos (i.e. not SpeechBrain or Huggingface).
"""

from speechbrain.integrations.models.sgmse_plus import ScoreModel
from speechbrain.integrations.models.sgmse_moe import ScoreModelMoE

__all__ = ["ScoreModel", "ScoreModelMoE"]
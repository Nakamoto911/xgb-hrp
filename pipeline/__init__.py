"""End-to-end quant investment pipeline.

Orchestration layer fusing the JM/XGBoost regime forecaster (vendor/xgboost)
with the HRP portfolio engine (vendor/hrp), adding a daily Risk Monitor and
a per-step evaluation framework. See SPEC.md for the full design.
"""

from pipeline.config import PipelineConfig

__all__ = ["PipelineConfig"]
__version__ = "0.1.0"

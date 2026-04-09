"""Probability models: base interfaces, trainers, and ensemble."""

from moneygone.models.base import ModelPrediction, ProbabilityModel
from moneygone.models.calibration import Calibrator
from moneygone.models.ensemble import EnsembleModel
from moneygone.models.evaluation import ModelEvaluator
from moneygone.models.registry import ModelRegistry
from moneygone.models.trainers.bayesian import BayesianModel
from moneygone.models.trainers.gbm import GBMModel
from moneygone.models.trainers.logistic import LogisticModel

__all__ = [
    # Base
    "ModelPrediction",
    "ProbabilityModel",
    # Evaluation & calibration
    "ModelEvaluator",
    "Calibrator",
    # Trainers
    "LogisticModel",
    "GBMModel",
    "BayesianModel",
    # Ensemble
    "EnsembleModel",
    # Registry
    "ModelRegistry",
]

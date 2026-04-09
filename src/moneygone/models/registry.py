"""Model registry for versioned model persistence and discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

import structlog

from moneygone.models.base import ProbabilityModel

log = structlog.get_logger()


class ModelRegistry:
    """Persistent registry for trained models.

    Stores models as joblib pickles alongside JSON metadata files in
    a directory structure::

        model_dir/
            logistic/
                v0.1.0/
                    model.joblib
                    metadata.json
                v0.2.0/
                    model.joblib
                    metadata.json
            gbm/
                v0.1.0/
                    ...

    Args:
        model_dir: Root directory for model storage.
    """

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model: ProbabilityModel, metadata: dict | None = None) -> Path:
        """Save a trained model to the registry with metadata.

        Args:
            model: A trained :class:`ProbabilityModel` instance.
            metadata: Optional dict of additional metadata (metrics,
                training info, etc.).

        Returns:
            Path to the directory where the model was saved.
        """
        version_dir = self._model_dir / model.name / model.version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.joblib"
        model.save(model_path)

        # Build and write metadata
        meta = {
            "model_name": model.name,
            "model_version": model.version,
            "trained_at": (
                model.trained_at.isoformat() if model.trained_at else None
            ),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            meta.update(metadata)

        meta_path = version_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        log.info(
            "model_registered",
            name=model.name,
            version=model.version,
            path=str(version_dir),
        )
        return version_dir

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, name: str, version: str) -> ProbabilityModel:
        """Load a model by name and version.

        Args:
            name: Model name (e.g., "logistic", "gbm").
            version: Version string (e.g., "0.1.0").

        Returns:
            A loaded :class:`ProbabilityModel` instance.

        Raises:
            FileNotFoundError: If the model/version doesn't exist.
        """
        model_path = self._model_dir / name / version / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model '{name}' version '{version}' not found at {model_path}"
            )

        # Determine the model class from the stored data
        model = self._load_model_from_path(model_path)
        log.info("model_loaded", name=name, version=version)
        return model

    def _load_model_from_path(self, path: Path) -> ProbabilityModel:
        """Load a model using the appropriate class loader.

        Inspects the joblib payload to determine the model type and
        dispatches to the correct class's ``load`` method.
        """
        # Peek at the data to determine model type
        data = joblib.load(path)

        # Try to determine model class from stored data or model type
        model_obj = data.get("model")
        if model_obj is not None:
            model_type = type(model_obj).__name__

            if "LogisticRegression" in model_type:
                from moneygone.models.trainers.logistic import LogisticModel
                return LogisticModel.load(path)
            elif "LGBM" in model_type:
                from moneygone.models.trainers.gbm import GBMModel
                return GBMModel.load(path)
            elif "BayesianRidge" in model_type:
                from moneygone.models.trainers.bayesian import BayesianModel
                return BayesianModel.load(path)

        # Fallback: check for scaler (indicates Bayesian)
        if "scaler" in data:
            from moneygone.models.trainers.bayesian import BayesianModel
            return BayesianModel.load(path)

        # Fallback: check for feature_importances (indicates GBM)
        if "feature_importances" in data:
            from moneygone.models.trainers.gbm import GBMModel
            return GBMModel.load(path)

        # Last resort: try logistic
        from moneygone.models.trainers.logistic import LogisticModel
        return LogisticModel.load(path)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """List all registered models with their metadata.

        Returns:
            List of dicts, each containing model metadata including
            ``model_name``, ``model_version``, and any custom metadata
            saved during registration.
        """
        models = []
        if not self._model_dir.exists():
            return models

        for model_dir in sorted(self._model_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for version_dir in sorted(model_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                meta_path = version_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    models.append(meta)
                else:
                    # Metadata file missing -- construct minimal entry
                    models.append(
                        {
                            "model_name": model_dir.name,
                            "model_version": version_dir.name,
                            "registered_at": None,
                        }
                    )

        return models

    def get_latest_version(self, name: str) -> str | None:
        """Get the latest registered version for a model name.

        Returns:
            Version string, or None if no versions are registered.
        """
        model_dir = self._model_dir / name
        if not model_dir.exists():
            return None

        versions = sorted(
            [d.name for d in model_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        return versions[0] if versions else None

    def delete(self, name: str, version: str) -> None:
        """Remove a model version from the registry.

        Args:
            name: Model name.
            version: Version string.
        """
        import shutil

        version_dir = self._model_dir / name / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            log.info("model_deleted", name=name, version=version)

            # Clean up empty parent
            model_dir = self._model_dir / name
            if model_dir.exists() and not any(model_dir.iterdir()):
                model_dir.rmdir()

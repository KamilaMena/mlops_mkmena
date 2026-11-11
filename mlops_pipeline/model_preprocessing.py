"""Módulo de compatibilidad.

La lógica de limpieza y transformación vive ahora en ``ft_engineering.py``.
Este archivo solo reexporta esos nombres para que sigan funcionando:

- los modelos ``.joblib`` ya guardados, que referencian
  ``model_preprocessing.CreditDataCleaner``;
- los módulos que todavía importan desde aquí (``model_monitoring``,
  ``model_deploy``).
"""

from mlops_pipeline.ft_engineering import (  # noqa: F401
    CATEGORICAL_COLUMNS,
    EXCLUDED_COLUMNS,
    MONETARY_COLUMNS,
    RARE_CREDIT_TYPES,
    TREND_CATEGORIES,
    CreditDataCleaner,
    build_preprocessing_pipeline,
    build_preprocessor,
)

__all__ = [
    'CATEGORICAL_COLUMNS',
    'EXCLUDED_COLUMNS',
    'MONETARY_COLUMNS',
    'RARE_CREDIT_TYPES',
    'TREND_CATEGORIES',
    'CreditDataCleaner',
    'build_preprocessing_pipeline',
    'build_preprocessor',
]
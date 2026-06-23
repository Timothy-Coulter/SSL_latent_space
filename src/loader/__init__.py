"""Dataset loaders."""

from .classification_datasets import get_classification_dataset
from .ssl_datasets import SSLPairDataset, get_ssl_dataset

__all__ = ["SSLPairDataset", "get_classification_dataset", "get_ssl_dataset"]

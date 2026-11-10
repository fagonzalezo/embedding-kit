"""EmbedKit — analyze and improve ML embedding spaces."""

__version__ = "0.2.0"

from embedkit.analysis.report import EmbedKitAnalyzer
from embedkit.api import EmbedKit

__all__ = ["EmbedKit", "EmbedKitAnalyzer", "__version__"]

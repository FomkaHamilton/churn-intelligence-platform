"""Abstract base class for all insight clients."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.insights.models import InsightData, InsightReport


class BaseInsightClient(ABC):
    """Common interface for insight generation backends."""

    @abstractmethod
    def generate(self, data: InsightData) -> InsightReport:
        """Produce a natural-language InsightReport from computed model results."""

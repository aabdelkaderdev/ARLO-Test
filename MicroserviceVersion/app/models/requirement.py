"""Requirement model - represents a software requirement."""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class MetricTrigger:
    """Represents a metric trigger condition."""
    metric: str = ""
    trigger: str = ""

    def __str__(self) -> str:
        return f"{self.metric}: {self.trigger}"


@dataclass
class Requirement:
    """Represents a software requirement with its parsed attributes."""
    _id_counter: int = field(default=0, init=False, repr=False)
    
    id: int = field(default=0)
    description: str = ""
    title: str = ""
    parsed: bool = False
    is_architecturally_significant: bool = False
    is_nfr: bool = False
    quality_attributes: List[str] = field(default_factory=list)
    condition_text: str = ""
    condition_embeddings: List[float] = field(default_factory=list)
    metric_triggers: List[MetricTrigger] = field(default_factory=list)
    condition_word_count: int = 0
    created_date: Optional[datetime] = None
    last_modified_date: Optional[datetime] = None

    def __post_init__(self):
        if self.id == 0:
            Requirement._id_counter += 1
            self.id = Requirement._id_counter

    def __str__(self) -> str:
        qa_str = ",".join(self.quality_attributes)
        return f"[{qa_str}]: {self.description}"

    def to_short_string(self) -> str:
        qa_str = ",".join(self.quality_attributes)
        return f"R{self.id}: [{qa_str}]"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "title": self.title,
            "parsed": self.parsed,
            "is_architecturally_significant": self.is_architecturally_significant,
            "quality_attributes": self.quality_attributes,
            "condition_text": self.condition_text,
        }

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter (useful for testing)."""
        cls._id_counter = 0

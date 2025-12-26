"""Decision model - represents an architectural decision."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Decision:
    """Represents an architectural pattern selection decision."""
    arch_pattern_name: str = ""
    selected_pattern: str = ""
    score: int = 0
    satisfied_qualities: List[Tuple[str, int]] = field(default_factory=list)
    unsatisfied_qualities: List[Tuple[str, int]] = field(default_factory=list)

    def __str__(self) -> str:
        if len(self.satisfied_qualities) + len(self.unsatisfied_qualities) == 0:
            return f"{self.selected_pattern} selected for {self.arch_pattern_name} without impacting any qualities."
        
        description = f"{self.selected_pattern} selected for {self.arch_pattern_name}."
        
        if self.satisfied_qualities:
            satisfied_str = "\n".join(
                f"-- {quality}: {score}" 
                for quality, score in self.satisfied_qualities
            )
            description += f"\n- satisfying\n {satisfied_str}"
        
        if self.unsatisfied_qualities:
            unsatisfied_str = "\n".join(
                f"-- {quality}: {score}" 
                for quality, score in self.unsatisfied_qualities
            )
            description += f"\n- Not satisfying\n {unsatisfied_str}"
        
        return description

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "arch_pattern_name": self.arch_pattern_name,
            "selected_pattern": self.selected_pattern,
            "score": self.score,
            "satisfied_qualities": [
                {"quality": q, "score": s} for q, s in self.satisfied_qualities
            ],
            "unsatisfied_qualities": [
                {"quality": q, "score": s} for q, s in self.unsatisfied_qualities
            ],
        }

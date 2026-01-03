"""Concern model - represents a set of decisions for satisfiable conditions."""
from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.decision import Decision
    from app.models.requirement import Requirement


@dataclass
class ConditionGroup:
    """Groups requirements with equivalent conditions."""
    nominal_condition: str = ""
    requirements: List["Requirement"] = field(default_factory=list)


@dataclass
class SatisfiableGroup:
    """Groups condition groups that can be true simultaneously."""
    condition_groups: List[ConditionGroup] = field(default_factory=list)


@dataclass
class Concern:
    """Represents a concern with decisions for a satisfiable group."""
    desired_qualities: Dict[str, int] = field(default_factory=dict)
    decisions: List["Decision"] = field(default_factory=list)
    satisfiable_group: SatisfiableGroup = field(default_factory=SatisfiableGroup)

    @property
    def conditions(self) -> List[str]:
        """Get list of nominal conditions."""
        return [cg.nominal_condition for cg in self.satisfiable_group.condition_groups]

    @property
    def average_score(self) -> float:
        """Calculate average decision score."""
        if not self.decisions:
            return 0.0
        return sum(d.score for d in self.decisions) / len(self.decisions)

    @property
    def total_score(self) -> int:
        """Calculate total decision score."""
        return sum(d.score for d in self.decisions)

    def __str__(self) -> str:
        conditions_str = "\n".join(self.conditions)
        qualities_str = ",".join(
            f"{k}:{v}" for k, v in sorted(
                self.desired_qualities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        )
        decisions_str = "\n".join(str(d) for d in self.decisions)
        
        return (
            f"Conditions:\n{conditions_str}\n\n"
            f"Desired Qualities:{qualities_str}\n\n"
            f"Average Decision Score (Max 100): {self.average_score:.2f}\n\n"
            f"Decisions:\n{decisions_str}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "conditions": self.conditions,
            "desired_qualities": self.desired_qualities,
            "average_score": self.average_score,
            "total_score": self.total_score,
            "decisions": [d.to_dict() for d in self.decisions],
        }

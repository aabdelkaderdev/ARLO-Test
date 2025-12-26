"""Reporting service - generates reports in Appendix B format."""
from typing import List, Dict, Optional
from datetime import datetime

from app.models.requirement import Requirement
from app.models.concern import Concern


class ReportingService:
    """Service for generating structured reports."""

    def __init__(self):
        self.report_lines: List[str] = []
        self.stats: Dict[str, float] = {}

    def clear(self):
        """Clear report and stats."""
        self.report_lines.clear()
        self.stats.clear()

    def writeline(self, text: str = "", echo: bool = True):
        """Add a line to the report."""
        self.report_lines.append(text)
        if echo:
            print(text)

    def record_stat(self, key: str, value: float):
        """Record a statistic."""
        self.stats[key] = value

    def generate_report(
        self,
        requirements: List[Requirement],
        asrs: List[Requirement],
        concerns: List[Concern],
        settings: Optional[dict] = None,
    ) -> str:
        """
        Generate a complete report in Appendix B format.
        
        Args:
            requirements: All parsed requirements
            asrs: Architecturally-significant requirements
            concerns: Generated concerns with decisions
            settings: Optional experiment settings
            
        Returns:
            Formatted report string
        """
        self.clear()
        
        # Header
        self.writeline("=" * 60)
        self.writeline("ARLO - Architectural Decision Report")
        self.writeline("=" * 60)
        self.writeline(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.writeline()
        
        # Settings
        if settings:
            self.writeline("Settings:")
            self.writeline("-" * 40)
            for key, value in settings.items():
                self.writeline(f"  {key}: {value}")
            self.writeline()
        
        # Requirements Summary
        self.writeline("Requirements Summary")
        self.writeline("-" * 40)
        self.writeline(f"  Total Requirements: {len(requirements)}")
        self.writeline(f"  Architecturally-Significant: {len(asrs)}")
        reqs_with_conditions = sum(
            1 for r in asrs 
            if r.condition_text and r.condition_text != "under any circumstances"
        )
        self.writeline(f"  With Conditions: {reqs_with_conditions}")
        self.writeline()
        
        # ASRs
        self.writeline("Architecturally-Significant Requirements (ASRs)")
        self.writeline("-" * 40)
        for req in asrs:
            self.writeline(f"\nR{req.id}: {req.description[:100]}...")
            self.writeline(f"  Quality Attributes: {', '.join(req.quality_attributes)}")
            if req.condition_text:
                self.writeline(f"  Condition: {req.condition_text}")
        self.writeline()
        
        # Concerns and Decisions
        self.writeline("=" * 60)
        self.writeline("Architectural Decisions")
        self.writeline("=" * 60)
        
        for i, concern in enumerate(concerns, 1):
            self.writeline()
            self.writeline(f"Concern {i}")
            self.writeline("-" * 40)
            
            # Conditions
            self.writeline("Conditions:")
            for condition in concern.conditions:
                self.writeline(f"  - {condition}")
            self.writeline()
            
            # Quality Weights
            self.writeline("Desired Qualities (weight):")
            sorted_qualities = sorted(
                concern.desired_qualities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for quality, weight in sorted_qualities:
                self.writeline(f"  - {quality}: {weight}")
            self.writeline()
            
            # Decisions
            self.writeline(f"Average Decision Score: {concern.average_score:.2f}")
            self.writeline()
            self.writeline("Decisions:")
            for decision in concern.decisions:
                self.writeline(f"\n  {decision.arch_pattern_name}:")
                self.writeline(f"    Selected: {decision.selected_pattern}")
                self.writeline(f"    Score: {decision.score}")
                
                if decision.satisfied_qualities:
                    sat_str = ", ".join(
                        f"{q}({s})" for q, s in decision.satisfied_qualities
                    )
                    self.writeline(f"    Satisfies: {sat_str}")
                
                if decision.unsatisfied_qualities:
                    unsat_str = ", ".join(
                        f"{q}({s})" for q, s in decision.unsatisfied_qualities
                    )
                    self.writeline(f"    Tradeoffs: {unsat_str}")
        
        # Statistics
        if self.stats:
            self.writeline()
            self.writeline("=" * 60)
            self.writeline("Statistics")
            self.writeline("-" * 40)
            for key, value in self.stats.items():
                self.writeline(f"  {key}: {value}")
        
        self.writeline()
        self.writeline("=" * 60)
        self.writeline("End of Report")
        self.writeline("=" * 60)
        
        return "\n".join(self.report_lines)

    def to_dict(self) -> dict:
        """Convert report to dictionary format."""
        return {
            "report": "\n".join(self.report_lines),
            "stats": self.stats,
        }

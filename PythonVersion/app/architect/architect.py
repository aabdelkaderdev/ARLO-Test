"""Architect - main orchestration logic for architectural decision making."""
import re
import json
from enum import Enum
from typing import List, Dict, Optional, Tuple

from app.models.requirement import Requirement
from app.models.decision import Decision
from app.models.concern import Concern, ConditionGroup, SatisfiableGroup
from app.models.matrix import Matrix
from app.services.ollama_service import OllamaService
from app.services.parser_service import RequirementParser
from app.services.clustering_service import ClusteringService
from app.services.optimizer_service import Optimizer, OptimizerMode
from app.services.reporting_service import ReportingService


class QualityWeightsMode(str, Enum):
    """How to determine quality attribute weights."""
    EQUALLY_IMPORTANT = "EquallyImportant"
    ALL_REQUIRED = "AllRequired"
    INFERRED = "Inferred"
    PROVIDED = "Provided"


class Architect:
    """
    Main orchestration class for architectural decision making.
    
    Coordinates requirement parsing, condition grouping, and optimization
    to produce architectural decisions.
    """

    def __init__(
        self,
        ollama_service: Optional[OllamaService] = None,
        matrix_path: Optional[str] = None,
    ):
        self.ollama_service = ollama_service or OllamaService()
        self.matrix = Matrix.load_from_csv(matrix_path) if matrix_path else Matrix.load_from_csv()
        self.clustering_service = ClusteringService()
        self.optimizer = Optimizer()
        self.reporting_service = ReportingService()
        
        self.requirements: List[Requirement] = []
        self.asrs: List[Requirement] = []
        self.condition_groups: List[ConditionGroup] = []
        self.satisfiable_groups: List[SatisfiableGroup] = []
        self.concerns: List[Concern] = []
        self.quality_weights: Dict[str, int] = {}

    async def analyze(
        self,
        requirements_text: str,
        optimization_mode: OptimizerMode = OptimizerMode.ILP,
        quality_weights_mode: QualityWeightsMode = QualityWeightsMode.INFERRED,
        provided_weights: Optional[Dict[str, int]] = None,
        strict_asr_selection: bool = False,
    ) -> Tuple[List[Concern], str]:
        """
        Main analysis method - parses requirements and generates decisions.
        
        Args:
            requirements_text: Newline-separated requirements
            optimization_mode: ILP or Greedy
            quality_weights_mode: How to determine weights
            provided_weights: Optional user-provided weights
            strict_asr_selection: Use stricter ASR criteria
            
        Returns:
            Tuple of (list of concerns, report text)
        """
        # Reset state
        self.concerns.clear()
        self.condition_groups.clear()
        self.satisfiable_groups.clear()
        self.quality_weights.clear()
        
        # Step 1: Parse requirements
        print(">> Parsing Requirements ...")
        parser = RequirementParser(self.ollama_service)
        parser.load_from_text(requirements_text)
        await parser.parse(strict_asr_selection)
        
        self.requirements = parser.requirements
        self.asrs = parser.get_asrs()
        
        if not self.asrs:
            report = self.reporting_service.generate_report(
                self.requirements, [], [], 
                {"message": "No architecturally-significant requirements found"}
            )
            return [], report
        
        # Step 2: Generate condition groups
        print(">> Generating Condition Groups ...")
        await self._generate_condition_groups()
        
        # Step 3: Generate satisfiable groups
        print(">> Generating Satisfiable Groups ...")
        await self._generate_satisfiable_groups()
        
        # Step 4: Generate decisions for each satisfiable group
        print(">> Generating Decisions ...")
        for sg in self.satisfiable_groups:
            # Calculate quality weights for this group
            self.quality_weights = self._calculate_quality_weights(
                sg, quality_weights_mode, provided_weights
            )
            
            # Run optimization
            decisions, satisfaction_scores = self.optimizer.optimize(
                optimization_mode,
                list(self.quality_weights.keys()),
                self.matrix,
                self._normalize_weights(self.quality_weights),
            )
            
            concern = Concern(
                satisfiable_group=sg,
                desired_qualities=dict(self.quality_weights),
                decisions=decisions,
            )
            self.concerns.append(concern)
        
        # Generate report
        settings = {
            "optimization_mode": optimization_mode.value,
            "quality_weights_mode": quality_weights_mode.value,
            "strict_asr_selection": strict_asr_selection,
        }
        report = self.reporting_service.generate_report(
            self.requirements, self.asrs, self.concerns, settings
        )
        
        return self.concerns, report

    async def _generate_condition_groups(self) -> None:
        """Group ASRs by equivalent conditions using embeddings and LLM."""
        # Separate ASRs with and without conditions
        any_condition = RequirementParser.ANY_CIRCUMSTANCES_CONDITION
        reqs_with_conditions = [
            r for r in self.asrs if r.condition_text != any_condition
        ]
        reqs_without_conditions = [
            r for r in self.asrs if r.condition_text == any_condition
        ]
        
        # Add "any circumstances" group
        if reqs_without_conditions:
            group = ConditionGroup(
                nominal_condition=any_condition,
                requirements=reqs_without_conditions,
            )
            self.condition_groups.append(group)
        
        if not reqs_with_conditions:
            return
        
        # Get embeddings for conditions
        conditions = [r.condition_text for r in reqs_with_conditions]
        embeddings = await self.ollama_service.get_embeddings(conditions)
        
        # Store embeddings
        for i, req in enumerate(reqs_with_conditions):
            if i < len(embeddings):
                req.condition_embeddings = embeddings[i]
        
        # Cluster conditions
        valid_embeddings = [e for e in embeddings if e]
        if len(valid_embeddings) < 2:
            # Not enough for clustering, each gets its own group
            for req in reqs_with_conditions:
                group = ConditionGroup(
                    nominal_condition=req.condition_text,
                    requirements=[req],
                )
                self.condition_groups.append(group)
            return
        
        cluster_assignments = self.clustering_service.cluster_conditions(valid_embeddings)
        cluster_map = ClusteringService.map_to_clusters(
            reqs_with_conditions, cluster_assignments
        )
        
        # For each cluster, use LLM to verify equivalence
        for cluster_reqs in cluster_map.values():
            cluster_groups = []
            
            for req in cluster_reqs:
                if not cluster_groups:
                    cluster_groups.append(ConditionGroup(
                        nominal_condition=req.condition_text,
                        requirements=[req],
                    ))
                else:
                    # Check equivalence with existing groups
                    found_equivalent = False
                    for group in cluster_groups:
                        is_equivalent = await self._check_condition_equivalence(
                            req.condition_text, group.nominal_condition
                        )
                        if is_equivalent:
                            group.requirements.append(req)
                            found_equivalent = True
                            break
                    
                    if not found_equivalent:
                        cluster_groups.append(ConditionGroup(
                            nominal_condition=req.condition_text,
                            requirements=[req],
                        ))
            
            self.condition_groups.extend(cluster_groups)

    async def _check_condition_equivalence(
        self, condition1: str, condition2: str
    ) -> bool:
        """Use LLM to check if two conditions are equivalent."""
        instructions = (
            "If the following conditions could mean the same thing or one can infer "
            "another or one can be considered a subset of another, return 'True' "
            "otherwise return 'False'. Just return True or False."
        )
        prompt = f"Condition 1: '{condition1}'\nCondition 2: '{condition2}'"
        
        try:
            response = await self.ollama_service.call(instructions, prompt)
            return "true" in response.lower()
        except Exception as e:
            print(f"Error checking equivalence: {e}")
            return False

    async def _generate_satisfiable_groups(self) -> None:
        """Use LLM to group conditions that can be true simultaneously."""
        if len(self.condition_groups) <= 1:
            # Single group, just use it
            if self.condition_groups:
                self.satisfiable_groups.append(SatisfiableGroup(
                    condition_groups=list(self.condition_groups)
                ))
            return
        
        instructions = (
            "Organize the provided set of conditions into groups where conditions "
            "in the same group can be true at the same time. Return the IDs of "
            "conditions in each group enclosed in parentheses. "
            "For example: ((1,2),(3,4)). "
            "If a condition is 'under any circumstances', include it in all groups. "
            "Return ONLY the ID format, no other text."
        )
        
        conditions_text = "\n".join(
            f"{i+1}: {cg.nominal_condition}" 
            for i, cg in enumerate(self.condition_groups)
        )
        prompt = f"Conditions:\n{conditions_text}"
        
        # Try parsing response
        for _ in range(3):  # Retry up to 3 times
            try:
                response = await self.ollama_service.call(instructions, prompt)
                self._parse_satisfiable_groups_response(response)
                if self.satisfiable_groups:
                    return
            except Exception as e:
                print(f"Error parsing satisfiable groups: {e}")
        
        # Fallback: treat all as one group
        self.satisfiable_groups.append(SatisfiableGroup(
            condition_groups=list(self.condition_groups)
        ))

    def _parse_satisfiable_groups_response(self, response: str) -> None:
        """Parse LLM response for satisfiable groups."""
        # Clean response
        response = response.strip().strip("()")
        
        # Split by groups: "),(" or "), ("
        groups = re.split(r"\)\s*,\s*\(", response)
        
        for group_str in groups:
            ids = []
            id_strings = group_str.split(",")
            
            for id_str in id_strings:
                id_str = id_str.strip().strip("()")
                try:
                    idx = int(id_str)
                    if 1 <= idx <= len(self.condition_groups):
                        ids.append(idx)
                except ValueError:
                    continue
            
            if ids:
                sg = SatisfiableGroup(
                    condition_groups=[
                        self.condition_groups[i - 1] for i in ids
                    ]
                )
                self.satisfiable_groups.append(sg)

    def _calculate_quality_weights(
        self,
        satisfiable_group: SatisfiableGroup,
        mode: QualityWeightsMode,
        provided_weights: Optional[Dict[str, int]],
    ) -> Dict[str, int]:
        """Calculate quality attribute weights based on mode."""
        weights = {}
        
        if mode == QualityWeightsMode.PROVIDED and provided_weights:
            return dict(provided_weights)
        
        if mode == QualityWeightsMode.EQUALLY_IMPORTANT:
            # Get all qualities from matrix
            for _, columns in self.matrix.get_rows():
                for quality in columns.keys():
                    weights[quality] = 1
                break
        else:
            # Inferred: count occurrences in requirements
            for cg in satisfiable_group.condition_groups:
                for req in cg.requirements:
                    for quality in req.quality_attributes:
                        weights[quality] = weights.get(quality, 0) + 1
        
        return weights

    def _normalize_weights(self, weights: Dict[str, int]) -> Dict[str, int]:
        """Normalize weights to percentages."""
        total = sum(weights.values())
        if total == 0:
            return weights
        
        return {k: (v * 100) // total for k, v in weights.items()}

    def get_results_summary(self) -> dict:
        """Get a summary of results for API response."""
        return {
            "total_requirements": len(self.requirements),
            "asr_count": len(self.asrs),
            "condition_groups": len(self.condition_groups),
            "satisfiable_groups": len(self.satisfiable_groups),
            "concerns": [c.to_dict() for c in self.concerns],
        }

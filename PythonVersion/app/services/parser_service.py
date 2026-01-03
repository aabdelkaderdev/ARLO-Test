"""Requirement parser service - parses requirements using LLM."""
import json
import time
from typing import List, Optional, Callable, TYPE_CHECKING
from app.models.requirement import Requirement

if TYPE_CHECKING:
    from app.services.llm_interface import LLMServiceInterface


# Type alias for progress callback
# callback(estimated_remaining_secs, current_index, total_count)
ProgressCallback = Callable[[float, int, int], None]


class RequirementParser:
    """Parses software requirements to identify ASRs and quality attributes."""
    
    BATCH_SIZE = 10
    ANY_CIRCUMSTANCES_CONDITION = "under any circumstances"
    MAX_BATCH_TIMES = 10  # Keep last N batch times for rolling average
    
    def __init__(self, llm_service: Optional['LLMServiceInterface'] = None):
        # Import here to avoid circular imports and allow any LLM service
        if llm_service is None:
            from app.services.ollama_service import OllamaService
            llm_service = OllamaService()
        self.llm_service = llm_service
        self.requirements: List[Requirement] = []
        self._batch_times: List[float] = []

    def load_from_text(self, text: str) -> None:
        """Load requirements from newline-separated text."""
        Requirement.reset_id_counter()
        self.requirements.clear()
        self._batch_times.clear()
        
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line:
                self.requirements.append(Requirement(description=line))

    async def parse(
        self, 
        only_select_absolutely_significant: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[Requirement]:
        """
        Parse requirements using LLM to identify ASRs and quality attributes.
        
        Args:
            only_select_absolutely_significant: If True, use stricter ASR criteria
            progress_callback: Optional callback for progress updates
                Signature: callback(estimated_remaining_secs, current_index, total_count)
            
        Returns:
            List of parsed requirements
        """
        if not self.requirements:
            return []
        
        print(f"Parsing {len(self.requirements)} Requirements ...")
        self._batch_times.clear()
        
        # Build instruction prompt
        instructions = self._build_instructions(only_select_absolutely_significant)
        
        index = 0
        while index < len(self.requirements):
            batch_start_time = time.time()
            
            batch_end = min(index + self.BATCH_SIZE, len(self.requirements))
            batch = self.requirements[index:batch_end]
            
            print(f"Parsing reqs {index + 1} to {batch_end} ...")
            
            # Build the prompt with requirements
            reqs_text = "\n".join(
                f"{r.id}. {r.description[:500]}"  # Truncate long descriptions
                for r in batch
            )
            
            try:
                response = await self.llm_service.call(instructions, reqs_text)
                parsed_reqs = self._parse_response(response)
                
                for parsed in parsed_reqs:
                    req = next(
                        (r for r in self.requirements if r.id == parsed.get("Id")),
                        None
                    )
                    if req:
                        req.condition_text = parsed.get("ConditionText", "")
                        req.is_architecturally_significant = parsed.get(
                            "IsArchitecturallySignificant", False
                        )
                        req.quality_attributes = parsed.get("QualityAttributes", [])
                        req.parsed = True
                        
                        if not req.condition_text or req.condition_text.upper() == "N/A":
                            req.condition_text = self.ANY_CIRCUMSTANCES_CONDITION
                
                asr_count = sum(1 for r in self.requirements if r.is_architecturally_significant)
                print(f"--> ASR Count (so far): {asr_count}")
                
            except Exception as e:
                print(f"!!! ERROR Parsing Req !!!\n{e}")
            
            # Track batch time for estimation
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            self._batch_times.append(batch_duration)
            
            # Keep only last N batch times
            if len(self._batch_times) > self.MAX_BATCH_TIMES:
                self._batch_times.pop(0)
            
            # Calculate estimated remaining time
            if progress_callback and self._batch_times:
                remaining_reqs = len(self.requirements) - batch_end
                remaining_batches = (remaining_reqs + self.BATCH_SIZE - 1) // self.BATCH_SIZE
                avg_batch_time = sum(self._batch_times) / len(self._batch_times)
                estimated_remaining = remaining_batches * avg_batch_time
                
                progress_callback(estimated_remaining, batch_end, len(self.requirements))
            
            index = batch_end
        
        return self.requirements

    def _build_instructions(self, strict: bool) -> str:
        """Build the instruction prompt for the LLM."""
        instructions = (
            "I have provided a set of software requirements. "
            "I want you to extract the following information and return a JSON array "
            "of the Requirement class provided below.\n"
        )
        
        if strict:
            instructions += (
                "1.Whether it is architecturally-significant. "
                "A requirement is Architecturally-significant if it satisfies both:\n"
                "1. It explicitly states a key decision regarding high-level software architecture.\n"
                "2. It specifies one or more of following quality attributes:\n"
            )
        else:
            instructions += (
                "Whether it is architecturally-significant. "
                "Architecturally-significant means specifying one or more of following "
                "quality attributes regarding overall software architecture:\n"
            )
        
        instructions += (
            "-Performance Efficiency: Achieving high performance under economic resource utilization.\n"
            "-Compatibility: Interoperability and co-existence.\n"
            "-Usability: A user-friendly app with straightforward and elegant UX and UI.\n"
            "-Reliability: Stability under different conditions.\n"
            "-Security: Protecting data, preventing breaches.\n"
            "-Maintainability: Easy to modify and improve.\n"
            "-Portability: Adaptable to different environments.\n"
            "-Cost Efficiency: Keep the overall cost as low as possible.\n\n"
            "2. Find the quality attributes mentioned from the list above.\n"
            "3. The ConditionText is a conditional statement provided in the requirement "
            "(e.g., 'if bandwidth is low', 'when traffic is high'). If none, return N/A.\n\n"
            "Return ONLY valid JSON array with this structure (no markdown, no explanation):\n"
            '[{"Id": 1, "IsArchitecturallySignificant": true, '
            '"QualityAttributes": ["Security"], "ConditionText": "N/A"}]'
        )
        
        return instructions

    def _parse_response(self, response: str) -> List[dict]:
        """Parse the LLM response as JSON."""
        # Clean up response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code block
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Try to find JSON array in response
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                return []
        
        return []

    def get_asrs(self) -> List[Requirement]:
        """Get only architecturally-significant requirements."""
        return [
            r for r in self.requirements 
            if r.parsed and r.is_architecturally_significant
        ]

    def get_parsing_stats(self) -> dict:
        """Get parsing statistics."""
        return {
            "total_requirements": len(self.requirements),
            "requirements_with_conditions": sum(
                1 for r in self.requirements 
                if r.condition_text != self.ANY_CIRCUMSTANCES_CONDITION
            ),
            "asr_count": sum(1 for r in self.requirements if r.is_architecturally_significant),
        }

"""Tests for requirement parser."""
import pytest
from app.models.requirement import Requirement
from app.services.parser_service import RequirementParser


class TestRequirementParser:
    """Test cases for RequirementParser class."""

    def test_load_from_text(self):
        """Test loading requirements from text."""
        parser = RequirementParser()
        
        text = """The system shall support 1000 users
The system must encrypt all data
The system should be easy to maintain"""
        
        parser.load_from_text(text)
        
        assert len(parser.requirements) == 3
        assert parser.requirements[0].description == "The system shall support 1000 users"
        assert parser.requirements[1].description == "The system must encrypt all data"
        assert parser.requirements[2].description == "The system should be easy to maintain"

    def test_load_from_text_empty_lines(self):
        """Test loading skips empty lines."""
        parser = RequirementParser()
        
        text = """Requirement 1

Requirement 2

"""
        
        parser.load_from_text(text)
        assert len(parser.requirements) == 2

    def test_id_assignment(self):
        """Test requirements get unique IDs."""
        Requirement.reset_id_counter()
        parser = RequirementParser()
        
        text = """Req1
Req2
Req3"""
        
        parser.load_from_text(text)
        
        ids = [r.id for r in parser.requirements]
        assert ids == [1, 2, 3]

    def test_get_asrs_empty_before_parse(self):
        """Test get_asrs returns empty before parsing."""
        parser = RequirementParser()
        parser.load_from_text("Some requirement")
        
        asrs = parser.get_asrs()
        assert len(asrs) == 0

    def test_build_instructions(self):
        """Test instruction building includes quality attributes."""
        parser = RequirementParser()
        
        instructions = parser._build_instructions(strict=False)
        
        assert "Performance Efficiency" in instructions
        assert "Security" in instructions
        assert "Maintainability" in instructions
        assert "JSON" in instructions

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        parser = RequirementParser()
        
        response = '[{"Id": 1, "IsArchitecturallySignificant": true, "QualityAttributes": ["Security"], "ConditionText": "N/A"}]'
        
        result = parser._parse_response(response)
        
        assert len(result) == 1
        assert result[0]["Id"] == 1
        assert result[0]["IsArchitecturallySignificant"] == True
        assert result[0]["QualityAttributes"] == ["Security"]

    def test_parse_response_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        parser = RequirementParser()
        
        response = """```json
[{"Id": 1, "IsArchitecturallySignificant": true, "QualityAttributes": [], "ConditionText": "N/A"}]
```"""
        
        result = parser._parse_response(response)
        assert len(result) == 1

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON returns empty list."""
        parser = RequirementParser()
        
        response = "This is not JSON"
        
        result = parser._parse_response(response)
        assert result == []


class TestRequirement:
    """Test cases for Requirement model."""

    def test_to_short_string(self):
        """Test short string representation."""
        Requirement.reset_id_counter()
        req = Requirement(
            description="Test requirement",
            quality_attributes=["Security", "Performance Efficiency"],
        )
        
        short = req.to_short_string()
        assert "R1:" in short
        assert "Security" in short
        assert "Performance Efficiency" in short

    def test_to_dict(self):
        """Test dictionary serialization."""
        req = Requirement(
            description="Test",
            quality_attributes=["Security"],
            condition_text="under normal load",
        )
        
        d = req.to_dict()
        
        assert d["description"] == "Test"
        assert d["quality_attributes"] == ["Security"]
        assert d["condition_text"] == "under normal load"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

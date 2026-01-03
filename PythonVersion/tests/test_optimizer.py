"""Tests for optimizer service."""
import pytest
from app.services.optimizer_service import Optimizer, OptimizerMode
from app.models.matrix import Matrix


class TestOptimizer:
    """Test cases for the Optimizer class."""

    @pytest.fixture
    def sample_matrix(self) -> Matrix:
        """Create a sample matrix for testing."""
        matrix = Matrix()
        
        # Add Deployment patterns
        matrix.row_groups["Monolith"] = "Deployment"
        matrix.set_element("Monolith", "Performance Efficiency", 1)
        matrix.set_element("Monolith", "Security", 1)
        matrix.set_element("Monolith", "Maintainability", -1)
        
        matrix.row_groups["Microservices"] = "Deployment"
        matrix.set_element("Microservices", "Performance Efficiency", 0)
        matrix.set_element("Microservices", "Security", 0)
        matrix.set_element("Microservices", "Maintainability", 1)
        
        # Add Database patterns
        matrix.row_groups["SQL"] = "Database Management"
        matrix.set_element("SQL", "Performance Efficiency", -1)
        matrix.set_element("SQL", "Security", -1)
        matrix.set_element("SQL", "Maintainability", 1)
        
        matrix.row_groups["NoSQL"] = "Database Management"
        matrix.set_element("NoSQL", "Performance Efficiency", 1)
        matrix.set_element("NoSQL", "Security", 1)
        matrix.set_element("NoSQL", "Maintainability", -1)
        
        return matrix

    def test_greedy_optimization(self, sample_matrix):
        """Test greedy optimization selects best patterns per group."""
        optimizer = Optimizer()
        
        desired_qualities = ["Performance Efficiency", "Security"]
        weights = {"Performance Efficiency": 50, "Security": 50}
        
        decisions, scores = optimizer.optimize(
            OptimizerMode.GREEDY,
            desired_qualities,
            sample_matrix,
            weights,
        )
        
        # Should have one decision per group
        assert len(decisions) == 2
        
        # Check deployment decision
        deployment = next(d for d in decisions if d.arch_pattern_name == "Deployment")
        assert deployment.selected_pattern == "Monolith"  # Better for perf/security
        
        # Check database decision
        database = next(d for d in decisions if d.arch_pattern_name == "Database Management")
        assert database.selected_pattern == "NoSQL"  # Better for perf/security

    def test_ilp_optimization(self, sample_matrix):
        """Test ILP optimization produces valid decisions."""
        optimizer = Optimizer()
        
        desired_qualities = ["Maintainability"]
        weights = {"Maintainability": 100}
        
        decisions, scores = optimizer.optimize(
            OptimizerMode.ILP,
            desired_qualities,
            sample_matrix,
            weights,
        )
        
        # Should have one decision per group
        assert len(decisions) == 2
        
        # Check deployment: Microservices is better for maintainability
        deployment = next(d for d in decisions if d.arch_pattern_name == "Deployment")
        assert deployment.selected_pattern == "Microservices"
        
        # Check database: SQL is better for maintainability
        database = next(d for d in decisions if d.arch_pattern_name == "Database Management")
        assert database.selected_pattern == "SQL"

    def test_empty_weights(self, sample_matrix):
        """Test optimization with empty weights."""
        optimizer = Optimizer()
        
        decisions, scores = optimizer.optimize(
            OptimizerMode.GREEDY,
            [],
            sample_matrix,
            {},
        )
        
        # Should still produce decisions (all with score 0)
        assert len(decisions) == 2


class TestMatrix:
    """Test cases for Matrix class."""

    def test_load_from_csv(self):
        """Test loading matrix from CSV file."""
        matrix = Matrix.load_from_csv()
        
        # Should have loaded groups
        groups = matrix.get_all_groups()
        assert len(groups) > 0
        
        # Check some expected groups exist
        assert "Deployment" in groups or any("Deploy" in g for g in groups)

    def test_get_rows_by_group(self):
        """Test getting rows by group."""
        matrix = Matrix()
        matrix.row_groups["Pattern1"] = "Group1"
        matrix.row_groups["Pattern2"] = "Group1"
        matrix.row_groups["Pattern3"] = "Group2"
        
        matrix.set_element("Pattern1", "Quality", 1)
        matrix.set_element("Pattern2", "Quality", 2)
        matrix.set_element("Pattern3", "Quality", 3)
        
        group1_rows = matrix.get_rows_by_group("Group1")
        assert len(group1_rows) == 2
        assert "Pattern1" in group1_rows
        assert "Pattern2" in group1_rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

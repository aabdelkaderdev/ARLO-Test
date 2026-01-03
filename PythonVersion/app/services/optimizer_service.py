"""Optimizer service - ILP and Greedy optimization for pattern selection."""
from enum import Enum
from typing import List, Dict, Tuple
from ortools.linear_solver import pywraplp

from app.models.decision import Decision
from app.models.matrix import Matrix


class OptimizerMode(str, Enum):
    """Optimization mode selection."""
    ILP = "ILP"
    GREEDY = "Greedy"


class Optimizer:
    """Optimizer for selecting architectural patterns using ILP or Greedy algorithms."""

    def optimize(
        self,
        mode: OptimizerMode,
        desired_qualities: List[str],
        matrix: Matrix,
        column_weights: Dict[str, int],
    ) -> Tuple[List[Decision], Dict[str, int]]:
        """
        Optimize pattern selection.
        
        Args:
            mode: Optimization mode (ILP or Greedy)
            desired_qualities: List of quality attributes to optimize for
            matrix: Quality-pattern matrix
            column_weights: Weights for each quality attribute
            
        Returns:
            Tuple of (list of decisions, satisfaction scores per quality)
        """
        if mode == OptimizerMode.ILP:
            return self._ilp(desired_qualities, matrix, column_weights)
        elif mode == OptimizerMode.GREEDY:
            return self._greedy(desired_qualities, matrix, column_weights)
        else:
            raise ValueError(f"Unsupported optimization mode: {mode}")

    def _greedy(
        self,
        desired_qualities: List[str],
        matrix: Matrix,
        column_weights: Dict[str, int],
    ) -> Tuple[List[Decision], Dict[str, int]]:
        """Greedy algorithm for pattern selection."""
        decisions = []
        
        # Get unique groups
        groups = matrix.get_all_groups()
        
        for group in groups:
            decision = Decision(arch_pattern_name=group, score=float("-inf"))
            decisions.append(decision)
            
            group_rows = matrix.get_rows_by_group(group)
            
            for pattern, columns in group_rows.items():
                satisfied = []
                unsatisfied = []
                row_value = 0
                
                for col_name, col_value in columns.items():
                    if col_name in desired_qualities:
                        weight = column_weights.get(col_name, 0)
                        row_value += col_value * weight
                        
                        if col_value > 0:
                            satisfied.append((col_name, col_value))
                        elif col_value < 0:
                            unsatisfied.append((col_name, col_value))
                
                if row_value > decision.score:
                    decision.score = row_value
                    decision.selected_pattern = pattern
                    decision.satisfied_qualities = satisfied
                    decision.unsatisfied_qualities = unsatisfied
        
        satisfaction_scores = self._calculate_satisfaction_scores(
            matrix, column_weights, decisions
        )
        
        return decisions, satisfaction_scores

    def _ilp(
        self,
        desired_qualities: List[str],
        matrix: Matrix,
        column_weights: Dict[str, int],
    ) -> Tuple[List[Decision], Dict[str, int]]:
        """Integer Linear Programming optimization."""
        # Create solver
        solver = pywraplp.Solver.CreateSolver("SCIP")
        
        if solver is None:
            print("Could not create solver.")
            return [], {}
        
        # Create binary variables for each pattern
        variables = {}
        for pattern, _ in matrix.get_rows():
            variables[pattern] = solver.IntVar(0, 1, pattern)
        
        # Add constraint: exactly one pattern per group
        groups = matrix.get_all_groups()
        for group in groups:
            group_rows = matrix.get_rows_by_group(group)
            constraint = solver.Constraint(1, 1, f"OnlyOneRowInGroup_{group}")
            for pattern in group_rows.keys():
                constraint.SetCoefficient(variables[pattern], 1)
        
        # Set objective: maximize weighted quality scores
        objective = solver.Objective()
        for pattern, columns in matrix.get_rows():
            row_score = sum(
                columns.get(q, 0) * column_weights.get(q, 0)
                for q in desired_qualities
            )
            objective.SetCoefficient(variables[pattern], row_score)
        
        objective.SetMaximization()
        
        # Solve
        status = solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            print("The problem does not have an optimal solution.")
            return [], {}
        
        # Extract decisions
        decisions = []
        for group in groups:
            group_rows = matrix.get_rows_by_group(group)
            
            for pattern, columns in group_rows.items():
                if variables[pattern].solution_value() == 1:
                    satisfied = []
                    unsatisfied = []
                    
                    for col_name, col_value in columns.items():
                        if col_name in desired_qualities:
                            if col_value > 0:
                                satisfied.append((col_name, col_value))
                            elif col_value < 0:
                                unsatisfied.append((col_name, col_value))
                    
                    score = sum(
                        columns.get(q, 0) * column_weights.get(q, 0)
                        for q in desired_qualities
                    )
                    
                    decision = Decision(
                        arch_pattern_name=group,
                        selected_pattern=pattern,
                        score=score,
                        satisfied_qualities=satisfied,
                        unsatisfied_qualities=unsatisfied,
                    )
                    decisions.append(decision)
                    break
        
        satisfaction_scores = self._calculate_satisfaction_scores(
            matrix, column_weights, decisions
        )
        
        return decisions, satisfaction_scores

    def _calculate_satisfaction_scores(
        self,
        matrix: Matrix,
        column_weights: Dict[str, int],
        decisions: List[Decision],
    ) -> Dict[str, int]:
        """Calculate overall satisfaction scores per quality attribute."""
        satisfaction_scores = {}
        
        for decision in decisions:
            group_rows = matrix.get_rows_by_group(decision.arch_pattern_name)
            selected_row = group_rows.get(decision.selected_pattern)
            
            if selected_row:
                for col_name, col_value in selected_row.items():
                    if col_name not in satisfaction_scores:
                        satisfaction_scores[col_name] = 0
                    
                    weight = column_weights.get(col_name, 0)
                    satisfaction_scores[col_name] += col_value * weight
        
        return satisfaction_scores

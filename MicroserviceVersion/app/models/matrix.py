"""Matrix model - represents the quality-architectural pattern matrix."""
from dataclasses import dataclass, field
from typing import Dict, Iterator, Tuple
import csv
import os


# Quality attribute abbreviation mappings
QA_MAPPINGS = {
    "PE": "Performance Efficiency",
    "CO": "Compatibility",
    "US": "Usability",
    "RE": "Reliability",
    "SE": "Security",
    "MA": "Maintainability",
    "PO": "Portability",
    "CE": "Cost Efficiency",
}


@dataclass
class Matrix:
    """Quality-Architectural Pattern Matrix for optimization."""
    _rows: Dict[str, Dict[str, int]] = field(default_factory=dict)
    row_groups: Dict[str, str] = field(default_factory=dict)

    def set_element(self, row_key: str, column_key: str, value: int) -> None:
        """Set an element in the matrix."""
        if row_key not in self._rows:
            self._rows[row_key] = {}
        self._rows[row_key][column_key] = value

    def get_element(self, row_key: str, column_key: str) -> int:
        """Get an element from the matrix."""
        if row_key in self._rows and column_key in self._rows[row_key]:
            return self._rows[row_key][column_key]
        raise KeyError(f"Row or Column key not found: {row_key}, {column_key}")

    def get_rows(self) -> Iterator[Tuple[str, Dict[str, int]]]:
        """Iterate over all rows."""
        for row_key, row_values in self._rows.items():
            yield row_key, row_values

    def get_rows_by_group(self, group: str) -> Dict[str, Dict[str, int]]:
        """Get all rows belonging to a specific group."""
        group_rows = {}
        for row_key, row_group in self.row_groups.items():
            if row_group == group:
                group_rows[row_key] = self._rows[row_key]
        return group_rows

    def get_all_groups(self) -> set:
        """Get all unique group names."""
        return set(self.row_groups.values())

    @classmethod
    def load_from_csv(
        cls, 
        file_path: str = None, 
        tab_separated: bool = True
    ) -> "Matrix":
        """Load matrix from CSV file."""
        if file_path is None:
            # Default to the matrix file in the data directory
            file_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "quality_archipattern_matrix_bal.csv"
            )
        
        matrix = cls()
        delimiter = "\t" if tab_separated else ","
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Parse header to get quality attribute names
        header_parts = [p.strip() for p in lines[0].split(delimiter) if p.strip()]
        # Map abbreviations to full names
        qualities = [QA_MAPPINGS.get(abbr, abbr) for abbr in header_parts]
        
        # Parse data rows
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(delimiter) if p.strip()]
            if len(parts) < 3:
                continue
            
            group = parts[0]
            pattern = parts[1]
            matrix.row_groups[pattern] = group
            
            # Parse quality values
            for i, value_str in enumerate(parts[2:]):
                if i < len(qualities):
                    try:
                        value = int(value_str)
                        matrix.set_element(pattern, qualities[i], value)
                    except ValueError:
                        continue
        
        return matrix

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rows": self._rows,
            "row_groups": self.row_groups,
        }

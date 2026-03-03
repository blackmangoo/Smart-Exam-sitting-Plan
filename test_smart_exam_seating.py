"""
Unit tests for SmartExamSeatingPlan
"""

import pytest
import pandas as pd
from smart_exam_seating import SmartExamSeatingPlan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_students():
    """12 students across 3 departments, 2 years, 2 sections."""
    data = [
        {"student_id": f"S{i:02d}", "department": dept, "year": yr, "section": sec}
        for i, (dept, yr, sec) in enumerate(
            [
                ("CS", 1, "A"), ("CS", 1, "A"), ("CS", 2, "B"),
                ("EE", 1, "A"), ("EE", 2, "A"), ("EE", 2, "B"),
                ("ME", 1, "B"), ("ME", 1, "B"), ("ME", 2, "A"),
                ("CE", 1, "A"), ("CE", 2, "A"), ("CE", 2, "B"),
            ],
            start=1,
        )
    ]
    return pd.DataFrame(data)


@pytest.fixture
def planner():
    return SmartExamSeatingPlan(n_clusters=3, random_state=0)


# ---------------------------------------------------------------------------
# cluster_students
# ---------------------------------------------------------------------------

class TestClusterStudents:
    def test_adds_cluster_column(self, planner, small_students):
        result = planner.cluster_students(small_students)
        assert "cluster" in result.columns

    def test_cluster_labels_are_integers(self, planner, small_students):
        result = planner.cluster_students(small_students)
        assert result["cluster"].dtype in (int, "int32", "int64")

    def test_cluster_count_does_not_exceed_n_clusters(self, planner, small_students):
        result = planner.cluster_students(small_students)
        assert result["cluster"].nunique() <= planner.n_clusters

    def test_auto_n_clusters_minimum_two(self, small_students):
        """When n_clusters is None it must be at least 2."""
        auto_planner = SmartExamSeatingPlan(n_clusters=None, random_state=0)
        result = auto_planner.cluster_students(small_students)
        assert result["cluster"].nunique() >= 2

    def test_original_df_unchanged(self, planner, small_students):
        original_cols = list(small_students.columns)
        planner.cluster_students(small_students)
        assert list(small_students.columns) == original_cols

    def test_single_student(self):
        df = pd.DataFrame([{"student_id": "X1", "department": "CS", "year": 1, "section": "A"}])
        p = SmartExamSeatingPlan(n_clusters=1, random_state=0)
        result = p.cluster_students(df)
        assert len(result) == 1
        assert result.iloc[0]["cluster"] == 0


# ---------------------------------------------------------------------------
# generate_seating_plan
# ---------------------------------------------------------------------------

class TestGenerateSeatingPlan:
    def test_grid_dimensions(self, planner, small_students):
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=5)
        assert len(grid) == 3
        assert all(len(row) == 5 for row in grid)

    def test_all_students_seated(self, planner, small_students):
        rows, cols = 3, 5
        grid, _ = planner.generate_seating_plan(small_students, rows=rows, cols=cols)
        seated = [seat for row in grid for seat in row if seat is not None]
        assert len(seated) == len(small_students)

    def test_no_duplicate_seats(self, planner, small_students):
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=5)
        ids = [seat["student_id"] for row in grid for seat in row if seat is not None]
        assert len(ids) == len(set(ids))

    def test_seat_coordinates_are_set(self, planner, small_students):
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=5)
        for r_idx, row in enumerate(grid):
            for c_idx, seat in enumerate(row):
                if seat is not None:
                    assert seat["row"] == r_idx + 1
                    assert seat["col"] == c_idx + 1

    def test_clustered_df_returned(self, planner, small_students):
        _, clustered = planner.generate_seating_plan(small_students, rows=3, cols=5)
        assert "cluster" in clustered.columns
        assert len(clustered) == len(small_students)

    def test_raises_when_too_few_seats(self, planner, small_students):
        with pytest.raises(ValueError, match="Not enough seats"):
            planner.generate_seating_plan(small_students, rows=2, cols=2)

    def test_exact_fit(self, planner, small_students):
        """Hall with exactly as many seats as students should have no empty seats."""
        n = len(small_students)
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=4)
        empty = sum(1 for row in grid for seat in row if seat is None)
        assert empty == 0


# ---------------------------------------------------------------------------
# malpractice_score
# ---------------------------------------------------------------------------

class TestMalpracticeScore:
    def test_score_between_zero_and_one(self, planner, small_students):
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=5)
        score = SmartExamSeatingPlan.malpractice_score(grid)
        assert 0.0 <= score <= 1.0

    def test_perfect_alternating_grid_score_zero(self):
        """A grid where no two adjacent seats share a cluster has score 0."""
        def _seat(cluster):
            return {"cluster": cluster}

        grid = [
            [_seat(0), _seat(1), _seat(0)],
            [_seat(1), _seat(0), _seat(1)],
        ]
        assert SmartExamSeatingPlan.malpractice_score(grid) == 0.0

    def test_all_same_cluster_score_one(self):
        """If every student is in the same cluster the score is 1."""
        def _seat():
            return {"cluster": 0}

        grid = [[_seat(), _seat()], [_seat(), _seat()]]
        assert SmartExamSeatingPlan.malpractice_score(grid) == 1.0

    def test_empty_grid_score_zero(self):
        assert SmartExamSeatingPlan.malpractice_score([]) == 0.0

    def test_single_seat_score_zero(self):
        grid = [[{"cluster": 0}]]
        assert SmartExamSeatingPlan.malpractice_score(grid) == 0.0


# ---------------------------------------------------------------------------
# export_to_csv
# ---------------------------------------------------------------------------

class TestExportToCsv:
    def test_csv_has_correct_row_count(self, planner, small_students, tmp_path):
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=5)
        out = tmp_path / "plan.csv"
        SmartExamSeatingPlan.export_to_csv(grid, str(out))
        df = pd.read_csv(out)
        assert len(df) == len(small_students)

    def test_csv_has_row_and_col_columns(self, planner, small_students, tmp_path):
        grid, _ = planner.generate_seating_plan(small_students, rows=3, cols=5)
        out = tmp_path / "plan.csv"
        SmartExamSeatingPlan.export_to_csv(grid, str(out))
        df = pd.read_csv(out)
        assert "row" in df.columns and "col" in df.columns


# ---------------------------------------------------------------------------
# Cluster quality: interleaving improves adjacency compared to naive order
# ---------------------------------------------------------------------------

class TestInterleaving:
    def test_interleaved_score_lower_or_equal_naive(self, small_students):
        """Interleaved placement should not be worse than sorted-by-cluster order."""
        planner = SmartExamSeatingPlan(n_clusters=3, random_state=0)
        grid_interleaved, clustered = planner.generate_seating_plan(
            small_students, rows=3, cols=5
        )
        score_interleaved = SmartExamSeatingPlan.malpractice_score(grid_interleaved)

        # Build a naive grid (sorted by cluster, not interleaved)
        sorted_students = clustered.sort_values("cluster").reset_index(drop=True)
        naive_grid = []
        idx = 0
        for r in range(3):
            row = []
            for c in range(5):
                if idx < len(sorted_students):
                    rec = sorted_students.iloc[idx].to_dict()
                    rec["row"] = r + 1
                    rec["col"] = c + 1
                    row.append(rec)
                else:
                    row.append(None)
                idx += 1
            naive_grid.append(row)
        score_naive = SmartExamSeatingPlan.malpractice_score(naive_grid)

        assert score_interleaved <= score_naive

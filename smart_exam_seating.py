"""
Smart Exam Seating Plan using K-Means Clustering
=================================================
Optimizes student seating arrangement during exams to minimize malpractice by
ensuring students who are similar (same department, year, section) are placed
as far apart as possible.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SmartExamSeatingPlan:
    """
    Uses K-Means Clustering to group students by similarity (department, year,
    section) and then interleaves the clusters across the exam hall so that
    adjacent seats always belong to different student groups.
    """

    FEATURE_COLS = ["department", "year", "section"]

    def __init__(self, n_clusters=None, random_state=42):
        """
        Parameters
        ----------
        n_clusters : int or None
            Number of clusters for K-Means.  When *None* the value is derived
            automatically as ``max(2, n_students // 10)``.
        random_state : int
            Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._fitted_clusters = None
        self._scaler = StandardScaler()
        self._label_encoders = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_students(self, students_df, feature_cols=None):
        """
        Assign each student a cluster label using K-Means.

        Parameters
        ----------
        students_df : pd.DataFrame
            Must contain at least the columns listed in *feature_cols*.
        feature_cols : list[str] or None
            Columns to use as features.  Defaults to
            ``["department", "year", "section"]``.

        Returns
        -------
        pd.DataFrame
            Copy of *students_df* with an extra ``"cluster"`` column.
        """
        if feature_cols is None:
            feature_cols = self.FEATURE_COLS

        X = self._encode_and_scale(students_df, feature_cols)

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = max(2, len(students_df) // 10)
        n_clusters = min(n_clusters, len(students_df))

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        result = students_df.copy()
        result["cluster"] = labels
        self._fitted_clusters = result
        return result

    def generate_seating_plan(self, students_df, rows, cols, feature_cols=None):
        """
        Build a seating grid that interleaves clusters so that neighbours are
        always from different student groups.

        Parameters
        ----------
        students_df : pd.DataFrame
            Student records.
        rows : int
            Number of rows in the exam hall.
        cols : int
            Number of columns in the exam hall.
        feature_cols : list[str] or None
            Columns to use for clustering (defaults to
            ``["department", "year", "section"]``).

        Returns
        -------
        seating_grid : list[list[dict | None]]
            2-D list (rows × cols) where each entry is either a student record
            dict (with added ``"row"``, ``"col"``, and ``"cluster"`` keys) or
            ``None`` for an empty seat.
        students_clustered : pd.DataFrame
            Student data with the ``"cluster"`` column appended.
        """
        n_students = len(students_df)
        n_seats = rows * cols
        if n_students > n_seats:
            raise ValueError(
                f"Not enough seats: {n_seats} available for {n_students} students."
            )

        students_clustered = self.cluster_students(students_df, feature_cols)
        ordered_indices = self._interleave_clusters(students_clustered)

        seating_grid = []
        seat_ptr = 0
        for r in range(rows):
            row_seats = []
            for c in range(cols):
                if seat_ptr < len(ordered_indices):
                    idx = ordered_indices[seat_ptr]
                    record = students_clustered.loc[idx].to_dict()
                    record["row"] = r + 1
                    record["col"] = c + 1
                    row_seats.append(record)
                else:
                    row_seats.append(None)
                seat_ptr += 1
            seating_grid.append(row_seats)

        return seating_grid, students_clustered

    @staticmethod
    def display_seating_plan(seating_grid, student_id_col="student_id"):
        """
        Print the seating grid to stdout in a readable table format.

        Parameters
        ----------
        seating_grid : list[list[dict | None]]
            Output from :meth:`generate_seating_plan`.
        student_id_col : str
            Column name to use as the cell label in the printed table.
        """
        col_width = 12
        rows = len(seating_grid)
        cols = len(seating_grid[0]) if rows else 0

        header = " " * 6 + "".join(f"Col {c + 1:<{col_width - 4}}" for c in range(cols))
        separator = "-" * len(header)
        print(separator)
        print(header)
        print(separator)

        for r, row in enumerate(seating_grid):
            row_label = f"Row {r + 1:<2} "
            cells = []
            for seat in row:
                if seat is None:
                    cells.append("[ EMPTY ]".center(col_width))
                else:
                    label = str(seat.get(student_id_col, "?"))
                    cells.append(f"[{label}]".center(col_width))
            print(row_label + "".join(cells))
        print(separator)

    @staticmethod
    def export_to_csv(seating_grid, filepath):
        """
        Save the seating plan to a CSV file.

        Parameters
        ----------
        seating_grid : list[list[dict | None]]
            Output from :meth:`generate_seating_plan`.
        filepath : str or Path
            Destination file path.
        """
        records = []
        for row in seating_grid:
            for seat in row:
                if seat is not None:
                    records.append(seat)
        pd.DataFrame(records).to_csv(filepath, index=False)

    @staticmethod
    def malpractice_score(seating_grid):
        """
        Compute a malpractice-risk score for the seating plan.

        The score is the fraction of horizontally or vertically adjacent seat
        pairs where both students belong to the *same* cluster.  Lower is
        better; 0.0 means no adjacent pair shares a cluster.

        Parameters
        ----------
        seating_grid : list[list[dict | None]]
            Output from :meth:`generate_seating_plan`.

        Returns
        -------
        float
            Value in [0, 1].
        """
        same = 0
        total = 0
        rows = len(seating_grid)
        cols = len(seating_grid[0]) if rows else 0

        def cluster_of(seat):
            return seat.get("cluster") if seat is not None else None

        for r in range(rows):
            for c in range(cols):
                seat = seating_grid[r][c]
                if seat is None:
                    continue
                # check right neighbour
                if c + 1 < cols:
                    nbr = seating_grid[r][c + 1]
                    if nbr is not None:
                        total += 1
                        if cluster_of(seat) == cluster_of(nbr):
                            same += 1
                # check bottom neighbour
                if r + 1 < rows:
                    nbr = seating_grid[r + 1][c]
                    if nbr is not None:
                        total += 1
                        if cluster_of(seat) == cluster_of(nbr):
                            same += 1

        return same / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_and_scale(self, df, feature_cols):
        """Label-encode categoricals then standardise all features."""
        data = df[feature_cols].copy()
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self._label_encoders[col] = le
        return self._scaler.fit_transform(data.to_numpy(dtype=float))

    @staticmethod
    def _interleave_clusters(students_clustered):
        """
        Return student indices ordered so that consecutive entries cycle
        through all clusters (round-robin).  This guarantees that when the
        indices are placed sequentially into the seat grid, neighbours along
        a row belong to different clusters wherever possible.
        """
        cluster_col = students_clustered["cluster"]
        n_clusters = cluster_col.nunique()

        buckets = []
        for cid in range(n_clusters):
            buckets.append(
                students_clustered.index[cluster_col == cid].tolist()
            )

        interleaved = []
        max_len = max(len(b) for b in buckets)
        for i in range(max_len):
            for bucket in buckets:
                if i < len(bucket):
                    interleaved.append(bucket[i])
        return interleaved

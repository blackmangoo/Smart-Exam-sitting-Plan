# Smart Exam Sitting Plan

A Python system that uses **K-Means Clustering** to generate optimised exam-hall
seating arrangements, minimising the risk of malpractice by ensuring students
who are similar (same department, year, or section) are placed as far apart as
possible.

---

## How It Works

1. **Feature extraction** — each student record carries attributes such as
   `department`, `year`, and `section`.
2. **K-Means Clustering** — students are grouped into *k* clusters based on
   their similarity.  Students in the same cluster are likely to know each
   other and share study material.
3. **Interleaved seat assignment** — clusters are round-robined across the exam
   hall so that horizontally and vertically adjacent seats always belong to
   *different* clusters wherever possible.
4. **Malpractice-risk score** — a metric in \[0, 1] measuring the fraction of
   adjacent seat-pairs that share a cluster.  0 = perfect separation, 1 = no
   separation.

---

## Project Structure

```
smart_exam_seating.py       # Core module — SmartExamSeatingPlan class
main.py                     # Demo / CLI entry-point with sample data
test_smart_exam_seating.py  # Unit tests (pytest)
requirements.txt            # Python dependencies
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
python main.py
```

Sample output:

```
============================================================
  Smart Exam Sitting Plan — K-Means Clustering
============================================================

Students  : 24
Hall size : 4 rows × 6 cols = 24 seats

--- Cluster Assignments ---
student_id   name department  year section  cluster
     CS103  Carol         CS     1       B        0
     ...

--- Seating Plan (by Student ID) ---
------------------------------------------------------------------------------
      Col 1       Col 2       Col 3       Col 4       Col 5       Col 6
------------------------------------------------------------------------------
Row 1    [CS103]     [CS101]     [CS106]     [EE104]  ...
...
------------------------------------------------------------------------------

Malpractice-risk score : 0.026  (0=best, 1=worst)
Seating plan saved to: seating_plan.csv
```

---

## API Usage

```python
import pandas as pd
from smart_exam_seating import SmartExamSeatingPlan

students = pd.DataFrame([
    {"student_id": "CS101", "department": "CS", "year": 1, "section": "A"},
    # ... more students
])

planner = SmartExamSeatingPlan(n_clusters=4, random_state=42)

# Generate a seating plan for a hall with 5 rows and 6 columns
seating_grid, students_clustered = planner.generate_seating_plan(
    students, rows=5, cols=6
)

# Print the seating plan
SmartExamSeatingPlan.display_seating_plan(seating_grid, student_id_col="student_id")

# Measure malpractice risk (lower is better)
score = SmartExamSeatingPlan.malpractice_score(seating_grid)

# Export to CSV
SmartExamSeatingPlan.export_to_csv(seating_grid, "seating_plan.csv")
```

### `SmartExamSeatingPlan` parameters

| Parameter      | Default | Description                                                      |
|----------------|---------|------------------------------------------------------------------|
| `n_clusters`   | `None`  | Number of K-Means clusters.  Auto-detected when `None`.         |
| `random_state` | `42`    | RNG seed for reproducible results.                               |

### Key methods

| Method                       | Description                                              |
|------------------------------|----------------------------------------------------------|
| `cluster_students(df)`       | Returns df with a `cluster` column added.               |
| `generate_seating_plan(...)` | Returns a 2-D seating grid and the clustered df.        |
| `display_seating_plan(...)`  | Prints a formatted hall layout to stdout.               |
| `malpractice_score(grid)`    | Computes the adjacency-cluster overlap score.           |
| `export_to_csv(grid, path)`  | Saves the seating plan to a CSV file.                   |

---

## Running Tests

```bash
pip install pytest
pytest test_smart_exam_seating.py -v
```

All 21 tests should pass.

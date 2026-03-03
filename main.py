"""
Smart Exam Seating Plan — Demo / CLI entry-point
=================================================
Run this script to see a sample seating arrangement generated with K-Means
Clustering for a fictional set of students.

Usage
-----
    python main.py
"""

import pandas as pd
from smart_exam_seating import SmartExamSeatingPlan


# ---------------------------------------------------------------------------
# Sample student data
# ---------------------------------------------------------------------------

SAMPLE_STUDENTS = [
    # CS department
    {"student_id": "CS101", "name": "Alice",    "department": "CS", "year": 1, "section": "A"},
    {"student_id": "CS102", "name": "Bob",      "department": "CS", "year": 1, "section": "A"},
    {"student_id": "CS103", "name": "Carol",    "department": "CS", "year": 1, "section": "B"},
    {"student_id": "CS104", "name": "Dave",     "department": "CS", "year": 2, "section": "A"},
    {"student_id": "CS105", "name": "Eve",      "department": "CS", "year": 2, "section": "B"},
    {"student_id": "CS106", "name": "Frank",    "department": "CS", "year": 3, "section": "A"},
    # EE department
    {"student_id": "EE101", "name": "Grace",    "department": "EE", "year": 1, "section": "A"},
    {"student_id": "EE102", "name": "Hank",     "department": "EE", "year": 1, "section": "B"},
    {"student_id": "EE103", "name": "Ivy",      "department": "EE", "year": 2, "section": "A"},
    {"student_id": "EE104", "name": "Jack",     "department": "EE", "year": 2, "section": "B"},
    {"student_id": "EE105", "name": "Kate",     "department": "EE", "year": 3, "section": "A"},
    {"student_id": "EE106", "name": "Leo",      "department": "EE", "year": 3, "section": "B"},
    # ME department
    {"student_id": "ME101", "name": "Mia",      "department": "ME", "year": 1, "section": "A"},
    {"student_id": "ME102", "name": "Noah",     "department": "ME", "year": 1, "section": "B"},
    {"student_id": "ME103", "name": "Olivia",   "department": "ME", "year": 2, "section": "A"},
    {"student_id": "ME104", "name": "Paul",     "department": "ME", "year": 2, "section": "B"},
    {"student_id": "ME105", "name": "Quinn",    "department": "ME", "year": 3, "section": "A"},
    {"student_id": "ME106", "name": "Rose",     "department": "ME", "year": 3, "section": "B"},
    # CE department
    {"student_id": "CE101", "name": "Sam",      "department": "CE", "year": 1, "section": "A"},
    {"student_id": "CE102", "name": "Tina",     "department": "CE", "year": 1, "section": "B"},
    {"student_id": "CE103", "name": "Uma",      "department": "CE", "year": 2, "section": "A"},
    {"student_id": "CE104", "name": "Victor",   "department": "CE", "year": 2, "section": "B"},
    {"student_id": "CE105", "name": "Wendy",    "department": "CE", "year": 3, "section": "A"},
    {"student_id": "CE106", "name": "Xander",   "department": "CE", "year": 3, "section": "B"},
]

# Exam-hall layout
ROWS = 4
COLS = 6


def main():
    students_df = pd.DataFrame(SAMPLE_STUDENTS)

    print("\n" + "=" * 60)
    print("  Smart Exam Seating Plan — K-Means Clustering")
    print("=" * 60)
    print(f"\nStudents  : {len(students_df)}")
    print(f"Hall size : {ROWS} rows × {COLS} cols = {ROWS * COLS} seats")

    planner = SmartExamSeatingPlan(n_clusters=4, random_state=42)

    seating_grid, students_clustered = planner.generate_seating_plan(
        students_df, rows=ROWS, cols=COLS
    )

    # Show cluster assignment
    print("\n--- Cluster Assignments ---")
    print(
        students_clustered[["student_id", "name", "department", "year", "section", "cluster"]]
        .sort_values("cluster")
        .to_string(index=False)
    )

    # Display seating plan
    print("\n--- Seating Plan (by Student ID) ---")
    SmartExamSeatingPlan.display_seating_plan(seating_grid, student_id_col="student_id")

    # Show malpractice risk score
    score = SmartExamSeatingPlan.malpractice_score(seating_grid)
    print(f"\nMalpractice-risk score : {score:.3f}  (0=best, 1=worst)")
    print("(Fraction of adjacent seat-pairs that share the same cluster)\n")

    # Export to CSV
    output_path = "seating_plan.csv"
    SmartExamSeatingPlan.export_to_csv(seating_grid, output_path)
    print(f"Seating plan saved to: {output_path}\n")


if __name__ == "__main__":
    main()

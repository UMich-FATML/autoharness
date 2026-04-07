import json
import os
from pathlib import Path

import numpy as np

from control_design.control_design import naive_greedy_schedule, s_sparse_greedy
from control_design.cost_function import fully_actuated_schedule, schedule_rank, trace_inverse_cost


TARGET_COST = 5.0
FULLY_ACTUATED_COST = 1.6663785652549696
RESULT_PATH = Path(os.environ.get("SPARSE_CONTROL_RESULT_PATH", "/app/results.json"))


def example1_system():
    A = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=float,
    )
    return A, B, 1, 5


def hidden_system():
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    return A, B, 1, 4


def schedule_to_list(schedule):
    return [list(slot) for slot in schedule]


def build_example1_report():
    A, B, sparsity, horizon = example1_system()
    schedule = s_sparse_greedy(A, B, sparsity, horizon)
    naive = naive_greedy_schedule(A, B, sparsity, horizon)
    report = {
        "schedule": schedule_to_list(schedule),
        "rank": schedule_rank(A, B, schedule, horizon),
        "example1_cost": trace_inverse_cost(A, B, schedule, horizon),
        "fully_actuated_cost": trace_inverse_cost(A, B, fully_actuated_schedule(B.shape[1], horizon), horizon),
        "naive_cost": trace_inverse_cost(A, B, naive, horizon),
    }
    return report


def write_example1_report(path=RESULT_PATH):
    report = build_example1_report()
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

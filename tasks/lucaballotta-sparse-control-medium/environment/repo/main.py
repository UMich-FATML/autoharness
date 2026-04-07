#!/usr/bin/env python3

from control_design.control_design import naive_greedy_schedule, s_sparse_greedy
from control_design.cost_function import fully_actuated_schedule, trace_inverse_cost
from pwc_tasks.task_common import example1_system


def main():
    A, B, sparsity, horizon = example1_system()
    greedy_schedule = s_sparse_greedy(A, B, sparsity, horizon)
    naive_schedule = naive_greedy_schedule(A, B, sparsity, horizon)
    full_schedule = fully_actuated_schedule(B.shape[1], horizon)

    print("fully actuated cost:", trace_inverse_cost(A, B, full_schedule, horizon))
    print("naive greedy cost:", trace_inverse_cost(A, B, naive_schedule, horizon))
    print("s-sparse greedy cost:", trace_inverse_cost(A, B, greedy_schedule, horizon))


if __name__ == "__main__":
    main()


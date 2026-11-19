"""Worker pools spawn their processes: forking a process that runs numba/OpenMP threads can crash the child."""

import os

from melodyai.parallel import process_pool


def _pid_and_square(x: int) -> tuple[int, int]:
    return os.getpid(), x * x


def test_process_pool_spawns_workers_and_runs_jobs():
    with process_pool(2) as pool:
        assert pool._mp_context.get_start_method() == "spawn"
        out = list(pool.map(_pid_and_square, range(4)))
    assert [sq for _, sq in out] == [0, 1, 4, 9]
    assert all(pid != os.getpid() for pid, _ in out)

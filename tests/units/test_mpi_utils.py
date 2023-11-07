import pytest
from ezpyp.utils import item_selection


@pytest.mark.parametrize("num_tasks", [2, 4, 8])
def test_exact_item_selection(num_tasks):
    items = list(range(num_tasks))
    for ii in range(num_tasks):
        assert item_selection(items, ii, num_tasks) == [ii]


@pytest.mark.parametrize("num_tasks", [2, 4, 8])
def test_dict_key_selection(num_tasks):
    items = {k: str(k) for k in range(num_tasks)}
    for ii in range(num_tasks):
        assert item_selection(items, ii, num_tasks) == [ii]


@pytest.mark.parametrize("rank,result", [[0, [0, 3]], [1, [1, 4]], [2, [2]]])
def test_drowned_task_availability(rank, result):
    n_items = 5
    mpi_size = 3
    tasks = list(range(n_items))
    assert item_selection(tasks, rank, mpi_size=mpi_size) == result


@pytest.mark.parametrize(
    "rank,result", [[0, [0]], [1, [1]], [2, [2]], [3, []], [4, []]]
)
def test_starved_task_availability(rank, result):
    n_items = 3
    mpi_size = 5
    tasks = list(range(n_items))
    assert item_selection(tasks, rank, mpi_size=mpi_size) == result

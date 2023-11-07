import pytest
from ezpyp.utils import item_selection


# @pytest.mark.parametrize("num_tasks", [2, 4, 8])
# def test_exact_item_selection(num_tasks):
#     items = list(range(num_tasks))
#
#     for ii in range(num_tasks):
#         assert item_selection(items, ii, num_tasks) == [ii]


def test_drowned_task_availability():
    n_items = 5
    mpi_size = 3

    tasks = list(range(n_items))

    for ii in range(mpi_size):
        assert item_selection(tasks, ii, mpi_size=mpi_size)[0] == ii

    assert item_selection(tasks, 0, mpi_size=mpi_size) == [0, 3]
    assert item_selection(tasks, 1, mpi_size=mpi_size) == [1, 4]


@pytest.mark.parametrize(
    "rank,result", [[0, [0]], [1, [1]], [2, [2]], [3, []], [4, []]]
)
def test_starved_task_availability(rank, result):
    n_items = 3
    mpi_size = 5

    tasks = list(range(n_items))

    assert item_selection(tasks, rank, mpi_size=mpi_size) == result

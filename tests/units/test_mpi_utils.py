import pytest
from ezpyp.utils import item_selection


@pytest.mark.parametrize("num_tasks", [2, 4, 8])
def test_exact_item_selection(num_tasks):
    num_tasks = 4
    items = list(range(num_tasks))
    for ii in range(num_tasks):
        assert item_selection(items, ii, num_tasks) == [ii]

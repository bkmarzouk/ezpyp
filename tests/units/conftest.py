import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
pytest-mpi has a similar fixture but requires usage of the
mpi marker. We want to run the test suite with and without
mpi in place, therefore want to avoid using the mpi marker
in most instances, otherwise we may need to copy tests.
"""


@pytest.fixture
def comm_tmp_path(tmp_path):
    if rank == 0:
        for ii in range(1, size):
            comm.send(tmp_path, dest=ii)
    else:
        tmp_path = comm.recv(source=0)

    return tmp_path

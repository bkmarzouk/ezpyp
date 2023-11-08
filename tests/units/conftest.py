import pytest
from ezpyp.mpi import MPI

COMM = MPI.COMM_WORLD
MPI_RANK = COMM.Get_rank()
MPI_SIZE = COMM.Get_size()

"""
pytest-mpi has a similar fixture but requires usage of the
mpi marker. We want to run the test suite with and without
mpi in place, therefore want to avoid using the mpi marker
in most instances, otherwise we may need to copy tests.
"""


@pytest.fixture
def comm_tmp_path(tmp_path):
    if MPI_RANK == 0:
        for ii in range(1, MPI_SIZE):
            COMM.send(tmp_path, dest=ii)
    else:
        tmp_path = COMM.recv(source=0)

    return tmp_path

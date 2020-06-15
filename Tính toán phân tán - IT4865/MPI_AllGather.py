# import numpy as np
# from mpi4py import MPI
# def rbind(comm, x):
#     return np.vstack(comm.allgather(x))
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# x = np.arange(4, dtype=np.int) + rank
# print("rank: ",rank)
# print("data: ",x)
# a = rbind(comm, x)
# print(a)

import numpy as np
from mpi4py import MPI
def rbind2(comm, x):
    size = comm.Get_size()
    m = np.zeros((size, len(x)), dtype=np.int)
    comm.Allgather([x, MPI.INT], [m, MPI.INT])
    return m
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
x = np.arange(4, dtype=np.int) * rank
print(rank)
a = rbind2(comm, x)
print(a)
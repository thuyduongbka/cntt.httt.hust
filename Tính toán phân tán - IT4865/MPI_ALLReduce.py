from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

value = np.arange(4, dtype=np.int) * rank

print(' Rank: ',rank, ' value = ', value)

size = comm.Get_size()
value_sum = np.zeros((size, len(value)), dtype=np.int)

comm.Allreduce([value, MPI.INT], [value_sum, MPI.INT],  op=MPI.SUM)
print(' Rank',rank, 'value_sum =    ',value_sum)

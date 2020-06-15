# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# if rank == 0:
#    data = [x for x in range(size)]
#    print ('we will be scattering:',data)
# else:
#    data = None
   
# data = comm.scatter(data, root=0)
# print ('rank',rank,'has data:',data)
		

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size() 
rank = comm.Get_rank()

numDataPerRank = 10  
data = None
if rank == 0:
    data = np.linspace(1,size*numDataPerRank,numDataPerRank*size)

recvbuf = np.empty(numDataPerRank, dtype='d') # allocate space for recvbuf
comm.Scatter(data, recvbuf, root=0)

print('Rank: ',rank, ', recvbuf received: ',recvbuf)

#the scatter takes an array of elements and distributes 
# the elements in the order of process rank
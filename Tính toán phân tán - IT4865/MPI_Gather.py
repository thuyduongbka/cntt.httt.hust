from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()   

numDataPerRank = 10  
sendbuf = np.linspace(rank*numDataPerRank+1,(rank+1)*numDataPerRank,numDataPerRank)
print('Rank: ',rank, ', sendbuf: ',sendbuf)

recvbuf = None
if rank == 0:
    recvbuf = np.empty(numDataPerRank*size, dtype='d')  

comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    print('Rank: ',rank, ', recvbuf received: ',recvbuf)

#Similar to scatter, gather takes elements from each process and 
# gathers them to the root process. 
# The elements are ordered by the rank of the process from which they were received.

#CÁCH 2
# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# if rank == 0:
#    data = [(x+1)**x for x in range(size)]
#    print ('we will be scattering:',data)
# else:
#    data = None
   
# data = comm.scatter(data, root=0)
# data += 1
# print ('rank',rank,'has data:',data)

# newData = comm.gather(data,root=0)

# if rank == 0:
#    print ('master:',newData)
		
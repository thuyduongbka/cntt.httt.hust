from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a_size = 1
senddata = (rank+1)*numpy.arange(size, dtype=int)
recvdata = numpy.empty(size*a_size, dtype=int)
comm.Alltoall(senddata, recvdata)

print("process %s sending %s receiving %s " % (rank,senddata,recvdata))

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    data = {'a':1,'b':2,'c':3}
else:
    data = None

data = comm.bcast(data, root=0)
print ('rank',rank,data)

#Khi tiến trình gốc (tiến trình 0) gọi Comm.Bcast, 
# biến dữ liệu sẽ được gửi đến tất cả các quy trình khác. 
# Khi tất cả các tiến trình nhận gọi Comm.Bcast, 
# biến dữ liệu sẽ được điền vào dữ liệu từ quy trình gốc.
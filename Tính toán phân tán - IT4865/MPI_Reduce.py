from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Create some np arrays on each process:
# For this demo, the arrays have only one
# entry that is assigned to be the rank of the processor
value = np.array(rank,'d')

print(' Rank: ',rank, ' value = ', value)

# initialize the np arrays that will store the results:
value_sum      = np.array(0.0,'d')
value_max      = np.array(0.0,'d')

# perform the reductions:
comm.Reduce(value, value_sum, op=MPI.SUM, root=0)
comm.Reduce(value, value_max, op=MPI.MAX, root=0)

if rank == 0:
    print(' Rank 0: value_sum =    ',value_sum)
    print(' Rank 0: value_max =    ',value_max)


#python comm.Reduce (send_data, recv_data, op =, root = 0) 
# send_data là dữ liệu được gửi từ tất cả các quy trình trên bộ truyền thông 
# recv_data là mảng trên tiến trình gốc sẽ nhận tất cả dữ liệu.  
# Một vài thao tác thường được sử dụng là: 
# - MPI_SUM - Tính tổng các phần tử. 
# - MPI_PROD - Nhân lên tất cả các yếu tố. 
# - MPI_MAX - Trả về phần tử tối đa. 
# - MPI_MIN - Trả về phần tử tối thiểu.
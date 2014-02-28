local r 

if mpi.get_rank() == 2 then
	r = "rank two"
end


r = mpi.bcast(2, r)

print(mpi.get_rank(), r)

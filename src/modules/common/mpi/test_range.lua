-- for k,v in pairs(mpi) do
-- 	print(k,v)
-- end


for i in mpi.range(1,10) do
	print(mpi.get_rank(), i)
end

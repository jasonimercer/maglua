-- rank = mpi.get_rank()
-- size = mpi.get_size()
-- 
-- if size ~= 8 then
-- 	error("Please run test with n = 8")
-- end
-- 
-- new_group = {1,1,1,2,2,1,5,2}
-- 
-- split_comm = mpi.comm_split(new_group[rank])
-- split_rank = split_comm:get_rank()
-- split_size = split_comm:get_size()
-- 
-- for i=1,size do
-- 	if rank == i then
-- 		print(rank .. "/" .. size .. " -> " .. split_rank .. "/" .. split_size)
-- 	end
-- 	mpi.barrier()
-- end




function f(x)
	print(x)
end

-- print(mpi.get_processor_name())

if mpi.get_rank() == 1 then
	t1 = Timer.new()
	t2 = Timer.new()
	t1:start()
	local t = {}
	for i=1,100 do
		t[i] = i
	end
	t2:start()
	mpi.send(2, t, f, "hello",t)
	t1:stop()
	t2:stop()
	print(t1:elapsed(), t2:elapsed())
else
	t,a,b = mpi.recv(1)
	a(b)
end
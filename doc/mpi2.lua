name = mpi.get_processor_name()
rank = mpi.get_rank()
size = mpi.get_size()

--if rank == 1 then
--	ss1 = SpinSystem.new(10,10,3)
--	mpi.send(2, ss1)
	print(name, rank, size)--, ss1)
--end

--if rank == 2 then
--	ss2 = mpi.recv(1)
--	print(name, rank, size, ss2)
--end

-- jason@echo: src$ mpirun -n 2 maglua_mpi mpi2.lua
-- echo    1       2       SpinSystem (10x10x3)
-- echo    2       2       SpinSystem (10x10x3)
-- jason@echo: src$ 

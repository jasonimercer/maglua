--
-- Lua MPI test 2
--

t = {}
rank = mpi.get_rank()
t[rank] = mpi.get_processor_name()

t = mpi.gather(1, t)

if rank == 1 then
	for i=1,mpi.get_size() do
		print(i, t[i])
	end
end


-- sample output:
-- $ mpirun --host cmms05,cmms04,cmms02,cmms01 -n 8 ./maglua_mpi mpi2.lua
-- 1	cmms05
-- 2	cmms04
-- 3	cmms02
-- 4	cmms01
-- 5	cmms05
-- 6	cmms04
-- 7	cmms02
-- 8	cmms01


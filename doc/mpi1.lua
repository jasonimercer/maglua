name = mpi.get_processor_name()
rank = mpi.get_rank()
size = mpi.get_size()

print(name, rank, size)

if rank == 1 then
	function f(x)
		print(x, x)
	end
	mpi.send(2, "hello", f)
end

if rank == 2 then
	word, func = mpi.recv(1)
	func(word)
end

-- jason@echo: src$ mpirun -n 4 maglua_mpi mpi1.lua
-- echo    1       4
-- echo    3       4
-- echo    4       4
-- echo    2       4
-- hello   hello
-- jason@echo: src$ 

if mpi.get_rank() == 1 then
	function f()
		local i = 0
		return function()
			i = i + 1
			return i
		end
	end
	
	local a = f()
	print(mpi.get_rank(), a())

	mpi.send(2, a)

	a = mpi.recv(2)
	print(mpi.get_rank(), a())
end


if mpi.get_rank() == 2 then
	local b = mpi.recv(1)
	print(mpi.get_rank(), b())
	mpi.send(1, b)
end

-- OUTPUT
-- $ mpirun -n 2 maglua -q test_upvalue.lua
-- 1       1
-- 2       2
-- 1       3



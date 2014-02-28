if mpi.get_rank() == 1 then
	local x,y,z = mpi.recv(2)

	print(x,y,z)
end


if mpi.get_rank() == 2 then
	local t = {4}
	mpi.send(1, t, t)
end


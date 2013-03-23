mpi.barrier()
if mpi.get_rank() == 1 then
	local t1 = Timer.new()
	local t2 = Timer.new()

	t1:start()
	t2:start()
	mpi.send(2, t2)
	t2 = mpi.recv(2)
	t1:stop()
	t2:stop()
	
	print(string.format("Timer dormant time during 2-way transfer: %f seconds", t1:elapsed() - t2:elapsed()))
else
	local tX = mpi.recv(1)
	mpi.send(1, tX)
end

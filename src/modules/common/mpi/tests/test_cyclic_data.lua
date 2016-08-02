if mpi.get_rank() == 1 then
	local t, q = {}, {}
	t[1] = 5
	t[2] = t
	q[1] = t

	mpi.send(2, t, q)
end

if mpi.get_rank() == 2 then
	local a, b = mpi.recv(1)

	b[1][1] = 6

	print(a[2][1], b[1][1])

end
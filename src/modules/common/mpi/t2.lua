-- tags ensure that different asyncronous communications don't clash
local tag = 5

if mpi.get_rank() == 2 then
	recv_request = mpi.irecv(1, tag)
end

mpi.barrier() -- barrier to show that an irecv can be before an isend

if mpi.get_rank() == 1 then
	local t = {}
	for i=1,1000 do
		table.insert(t, {1,2,3})
	end
	send_request = mpi.isend(2, tag, "hello", 5,6,7, t) 
	print("sent")
end

if mpi.get_rank() == 2 then
	print(recv_request:data()) -- not guaranteed to print data
	recv_request:wait()
	print(recv_request:data()) -- guaranteed to print "hello   5       6       7"
end


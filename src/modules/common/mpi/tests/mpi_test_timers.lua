-- This script demonstates that Timers are Encodable items (r205 and grater)
-- A Timer is created in rank == 1, clones are broadcasted, used and gathered

rank = mpi.get_rank()

if rank == 1 then
	t = Timer.new()
	t:start()
end

t = mpi.bcast(1, t) --broadcasting a live, running timer

for i=1,25000*rank do
	x = math.sin(i)
end
t:stop() --stop timers after calculations

tt = mpi.gather(1, t)

if rank == 1 then
	for k,v in pairs(tt) do
		print(k, v:elapsed())
	end
end

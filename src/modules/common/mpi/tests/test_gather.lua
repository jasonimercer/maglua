
local x
if mpi.get_rank() == 1 then
	x = "abc"
end

if mpi.get_rank() == 2 then
	x = 123
end


x = mpi.gather(1, x)

if x then
	for k,v in pairs(x) do
		print(k,v)
	end
end

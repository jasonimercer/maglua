-- dofile("info_diffusion.lua")
dofile("info_broadcast.lua")

macroX, macroY, macroZ  = 4, 4, 4
microX, microY, microZ = 16,16,16

ssMacro = SpinSystem.new(macroX, macroY, macroZ)
ssMicro = SpinSystem.new(microX, microY, microZ)

for k=1,microZ do
	for j=1,microY do
		for i=1,microX do
			ssMicro:setSpin({i,j,k}, {1,0,0}, 1)
		end
	end
end

setupCommunication(ssMacro)

-- faking calculations
for i=1,10 do
	ssMicro:setTime(ssMicro:time() + 0.1)
	communicate(ssMicro, ssMacro)
end

t1 = Timer.new()
t1:start()
ssMicro:setTime(ssMicro:time() + 0.1)
communicate(ssMicro, ssMacro)
t1:stop()

all_times = mpi.gather(1, t1:elapsed())

if mpi.get_rank() == 1 then
	local s, n = 0, 0
	for i=1,table.maxn(all_times) do
		s = s + all_times[i]
		n = n + 1
	end
	
	print("average communication time: " .. s/n)
end


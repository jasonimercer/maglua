local rank2position = {}
local macroX, macroY, macroZ

function setupCommunication(ssMacro)
	macroX, macroY, macroZ = ssMacro:nx(), ssMacro:ny(), ssMacro:nz()

	if mpi.get_size() ~= macroX * macroY * macroZ then
		error("need " .. macroX * macroY * macroZ .. " processes")
	end
	
	-- the following table maps between an mpi_rank and a position
	-- in the macro system
	rank2position = {}
	for k=1,macroZ do
		for j=1,macroY do
			for i=1,macroX do
				table.insert(rank2position, {i,j,k})
			end
		end
	end
	
end

function communicate(ssMicro, ssMacro)
	-- first we will update our local view of the system
	-- with the ssMicro's netMoment
	
	local x, y, z = ssMicro:netMoment()
	
	local all_spins = mpi.gather(1, {x,y,z})
	all_spins = mpi.bcast(1, all_spins)
	
	-- now that everyone has all the spin data, they can 
	-- all write to their local ssMacro
	
	for i=1,table.maxn(all_spins) do
		local pos = rank2position[i]
		local moment = all_spins[i]
		
		ssMacro:setSpin(pos, moment)
	end	
end








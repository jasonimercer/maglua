local macroX, macroY, macroZ
local rank2position = {}

local me --my position
local oddSite --used for communication scheduling

local neighbourX, rank_neighbourX
local neighbourx, rank_neighbourx
local neighbourY, rank_neighbourY
local neighboury, rank_neighboury
local neighbourZ, rank_neighbourZ
local neighbourz, rank_neighbourz

-- this is the local view of the system, not
-- all information will be from same time
local myMacroView = {}

-- this function sets up communication related structures
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
	-- lookup function to map from position to mpi_rank
	function position2rank(pos)
		for r,p in pairs(rank2position) do
			if p[1] == pos[1] and p[2] == pos[2] and p[3] == pos[3] then
				return r
			end
		end
	end

	-- figure out where we are and where our neighbours are
	me = rank2position[mpi.get_rank()]

	-- used to coordinate who speaks to who and when
	oddSite = math.mod(me[1]+me[2]+me[3],2) == 1


	-- convention: big X,Y,Z from +1, little x,y,z for -1
	neighbourX = {math.mod(me[1]-1 + 1, macroX)+1, me[2], me[3]}
	neighbourx = {math.mod(me[1]-1 - 1 + macroX, macroX)+1, me[2], me[3]}
	neighbourY = {me[1], math.mod(me[2]-1 + 1, macroY)+1, me[3]}
	neighboury = {me[1], math.mod(me[2]-1 - 1 + macroY, macroY)+1, me[3]}
	neighbourZ = {me[1], me[2], math.mod(me[3]-1 + 1, macroZ)+1}
	neighbourz = {me[1], me[2], math.mod(me[3]-1 - 1 + macroZ, macroZ)+1}

	-- testing
	-- if mpi.get_rank() == 4 then
	-- 	print("me", table.concat(me, ","))
	-- 	print("nX", table.concat(neighbourX, ","))
	-- 	print("nx", table.concat(neighbourx, ","))
	-- 	print("nY", table.concat(neighbourY, ","))
	-- 	print("ny", table.concat(neighboury, ","))
	-- 	print("nZ", table.concat(neighbourZ, ","))
	-- 	print("nz", table.concat(neighbourz, ","))
	-- end

	-- convert to ranks
	rank_neighbourX = position2rank(neighbourX)
	rank_neighbourx = position2rank(neighbourx)
	rank_neighbourY = position2rank(neighbourY)
	rank_neighboury = position2rank(neighboury)
	rank_neighbourZ = position2rank(neighbourZ)
	rank_neighbourz = position2rank(neighbourz)

	-- each process will have it's own idea about what the macro system looks like
	-- initially it'll be all zero vectors at time = 0. This could be improved
	-- with a single system-wide gather/bcast to start everyone with good information
	myMacroView = {}
	local zeroTime = 0
	for macroSite=1,macroX * macroY * macroZ do
		myMacroView[macroSite] = {zeroTime, 0,0,0}
	end
end

function communicate(ssMicro, ssMacro)
	-- first we will update our local view of the system
	-- with the ssMicro's netMoment
	
	local x, y, z = ssMicro:netMoment()
	myMacroView[mpi.get_rank()] = {ssMicro:time(), x, y, z}
	
	-- now we will communicate with the neighbours getting their
	-- views of the macro system
	
	local othersMacroView = {}
	-- odd rank processes talk first
	if oddSite then
		-- first we will communicate in the X direction
		-- odd will send to X+1
		mpi.send(rank_neighbourX, myMacroView)
		-- and then recv from X+1
		othersMacroView[1] = mpi.recv(rank_neighbourX)

		mpi.send(rank_neighbourx, myMacroView)
		othersMacroView[2] = mpi.recv(rank_neighbourx)

		-- next we will communicate in the Y direction
		mpi.send(rank_neighbourY, myMacroView)
		othersMacroView[3] = mpi.recv(rank_neighbourY)

		mpi.send(rank_neighboury, myMacroView)
		othersMacroView[4] = mpi.recv(rank_neighboury)

		-- finally we will communicate in the Y direction
		mpi.send(rank_neighbourZ, myMacroView)
		othersMacroView[5] = mpi.recv(rank_neighbourZ)

		mpi.send(rank_neighbourz, myMacroView)
		othersMacroView[6] = mpi.recv(rank_neighbourz)
	else
		-- even will recv from X-1
		othersMacroView[2] = mpi.recv(rank_neighbourx)
		-- and then send to X-1
		mpi.send(rank_neighbourx, myMacroView)

		othersMacroView[1] = mpi.recv(rank_neighbourX)
		mpi.send(rank_neighbourX, myMacroView)

		othersMacroView[4] = mpi.recv(rank_neighboury)
		mpi.send(rank_neighboury, myMacroView)

		othersMacroView[3] = mpi.recv(rank_neighbourY)
		mpi.send(rank_neighbourY, myMacroView)

		othersMacroView[6] = mpi.recv(rank_neighbourz)
		mpi.send(rank_neighbourz, myMacroView)

		othersMacroView[5] = mpi.recv(rank_neighbourZ)
		mpi.send(rank_neighbourZ, myMacroView)
	end
	
	-- now we have our view of the macroSystem and the
	-- views from the perspective of our neighbours
	-- we will combine these views taking the most recent data 
	-- for each site
	for macroSite=1,macroX * macroY * macroZ do
		for n=1,6 do
			-- if neighbour time is greater than our local time for the site
			if othersMacroView[n][macroSite][1] > myMacroView[macroSite][1] then
				myMacroView[macroSite] = othersMacroView[n][macroSite]
			end
		end
	end
	
	-- now that we have a new view of the macroSystem, we need to write it into 
	-- the ssMacro
	for macroSite=1,macroX * macroY * macroZ do
		local txyz = myMacroView[macroSite]
		ssMacro:setSpin(rank2position[macroSite], {txyz[2], txyz[3], txyz[4]})
	end
end








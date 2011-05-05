-- This script sets up and simulates a spherical particle.
-- See Byron Southern's code and Ken Adebayo's thesis.
--
-- This simulation cools from high temp to zero and then start
-- running loops at increasing temp


dofile("SetupSphericalParticle.lua") -- load in setup function
dofile("Magnetization.lua") -- get magnetization functions

if arg[1] == nil then
	error("Must supply L as first command line argument")
end
if arg[2] == nil then
	error("Must supply rate as second command line argument")
end

print(info()) --print info about version
rate = tonumber(arg[2])


L   = tonumber(arg[1])
ST  = 8
rng = Random.new("Isaac")

-- setup system, objects, etc
ss, ex, ani, regions, sinfo = makeSphericalParticle(L, ST, rng)
-- k1 to k4 for rk4
k1 = ss:copy()
k2 = ss:copy()
k3 = ss:copy()
k4 = ss:copy()

odt = ss:timeStep() --original timestep

print(sinfo)

th  = Thermal.new(ss:nx(), ss:ny(), ss:nz())
af  = AppliedField.new(ss:nx(), ss:ny(), ss:nz())
llg = LLG.new("Quaternion")


-- this function will advance the system for a time dt
-- executing the function func (or a dummy, empty function)
-- every r time. The arguments func and r are optional.
function run(dt, func, r)
	-- Calculate Fields given a spin system
	local function fields(spinsystem)
		spinsystem:zeroFields()
		
		ani:apply(spinsystem)
		ex:apply(spinsystem)
		af:apply(spinsystem)
		
		spinsystem:sumFields()
	end
	
	-- x = x or v is a way to give default args when they
	-- aren't passed in (and are therefore nil/false)
	func = func or function() end
	r    =    r or dt
	
	local next_func_call = ss:time() + r
	local t = ss:time()
	while ss:time() <=  t + dt do
		print(ss:time(), next_func_call, t+dt)
		-- rk4 update
		ss:copyTo(k1)
		fields(k1) -- fields at time t

		--llg has a special form where we can specify:
		-- src spins, src fields and dest spins
		ss:setTimeStep(odt/2)
		llg:apply(ss, k1, k2)
		fields(k2) --fields at t+dt/2 using k1
		
		--ss:setTimeStep(dt/2)
		llg:apply(ss, k2, k3)
		fields(k3) --fields at t+dt/2 using k2
		
		ss:setTimeStep(odt)
		llg:apply(ss, k3, k4)
		fields(k4) --fields at t+dt   using k3
		
		-- y(n+1) = y(n) + 1/6(k1 + 2k2 + 2k3 + k4)
		ss:zeroFields()
		ss:addFields(1/6, k1)
		ss:addFields(2/6, k2)
		ss:addFields(2/6, k3)
		ss:addFields(1/6, k4)
		
		llg:apply(ss)
		
		-- apply noise in a non-advancing llg step
		ss:zeroFields()
		th:apply(rng, ss)
		ss:sumFields()
		llg:apply(ss, false) --false does not advance time
		
		if ss:time() >= next_func_call then
			func()
			next_func_call = ss:time() + r
		end
	end
end


cfn = string.format("CoolingMT_%02iL_%05.3fR.dat", L, rate)

f = io.open(cfn, "w")
header = {}
for k,v in pairs(regions) do
	table.insert(header, k)
end

table.sort(header)

f:write(info("# ") .. "\n")
f:write(sinfo .. "\n#\n")
f:write("# temp time hz " .. table.concat(header, " ") .. "\n")
f:flush()

-- first, cool from T=100 to T=0 collecting magnetization stats
-- to compare against previous runs
af:set({0,0,0})
function runCoolingTemp(T)
	print("Temperature="..T)
	th:setTemperature(T)
	run(0.7)
	resetMagStats()
	run(0.3, collectMagStats, 0.05)
	
	local stats = calculateMagStats()
	
	-- prefixing mag data with temp, time and field 
	f:write(th:temperature() .. "\t")
	f:write(ss:time() .. "\t")
	f:write(af:get()[3] .. "\t")
	reportMagStats(stats, header, f)
	f:flush()
end

-- COOLING MAIN LOOP
for T=100,10,-5 do
	runCoolingTemp(T)
end
for T=9.5,0,-0.5 do
	runCoolingTemp(T)
end

f:close()
error() -- ending prematurely, need to get good netmag first

function doLoop(T)
	fn = string.format("MH_%05.1fT_%02iL_%05.3fR.dat", T, L, rate)
	print(fn)
	g = io.open(fn, "w")
	g:write(info("# ") .. "\n")
	g:write(sinfo .. "\n#\n")
	g:write("#" .. table.concat(header, "\t") .. "\n")
	local hmin, hmax, step = -1000, 1000, 20

	th:setTemperature(T)

	local function field_val(g, H, step)
		af:set(0,0,H)
		local time_per_field = 1.0 / rate
		local dt = time_per_field * step
		
-- 		run(0.7 * dt * 0.5)
		resetMagStats()

		run(0.3 * dt * 0.1, collectMagStats, 0.03 * dt * 0.1)

		local stats = calculateMagStats()

		-- prefixing mag data with temp, time and field and 
		g:write(th:temperature() .. "\t")
		g:write(ss:time() .. "\t")
		g:write(af:get()[3] .. "\t")
		reportMagStats(stats, cols, g)
		g:flush()
		print(dt,H)
	end
	
	-- rate = field/time
	-- lets say we want to go from -20 to 20 in time=80
	-- that means rate = 0.5
	-- we will step field by 0.5H. equilibriate for 70% and
	-- record magnetization for 30% over 10 states. Lets try this
	for H=hmin,hmax,step do
		field_val(g, H, step)
	end
	for H=hmax-step,hmin,-step do
		field_val(g, H, step)
	end
	
	
	g:close()

end


-- now we will start running loops and warming system between loops
for T=0.5,5,0.5 do
	doLoop(T)
end
for T=6,10 do
	doLoop(T)
end
for T=15,100,5 do
	doLoop(T)
end


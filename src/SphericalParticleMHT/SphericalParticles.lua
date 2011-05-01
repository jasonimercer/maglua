-- This script sets up and simulates a spherical particle.
-- See Byron Southern's code and Ken Adebayo's thesis.

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
ss, ex, ani, regions, info = makeSphericalParticle(L, ST, rng)
print(info)

th  = Thermal.new(ss:nx(), ss:ny(), ss:nz())
af  = AppliedField.new(ss:nx(), ss:ny(), ss:nz())
llg = LLG.new("Quaternion")


-- this function will advance the system for a time dt
-- executing the function func (or a dummy, empty function)
-- every r time. The arguments func and r are optional.
function run(dt, func, r)
	-- x = x or v is a way to give default args when they
	-- aren't passed in (and are therefore nil/false)
	func = func or function() end
	r    =    r or dt
	
	local next_func_call = ss:time() + r
	local t = ss:time()
	while ss:time() <=  t + dt do
		-- standard field/update calc
		ss:zeroFields()
		
		ani:apply(ss)
		ex:apply(ss)
		af:apply(ss)
		th:apply(rng, ss)
		
		ss:sumFields()
		
		llg:apply(ss)
		
		if ss:time() >= next_func_call then
			func()
			next_func_call = ss:time() + r
		end
	end
end


cfn = string.format("CoolingMT_%02iL_%05.3fR.dat", L, rate)

f = io.open(cfn, "w")
cols = {"total", "surf", "core"}
header = {"temp", "time", "hz"}
for k1,v1 in pairs({"total", "surf", "core"}) do
	for k2,v2 in pairs({"x", "y", "z"}) do
		table.insert(header, v1 .. "_" .. v2)
	end
end
f:write("#" .. table.concat(header, "\t") .. "\n")

-- if false then
-- first, cool from T=50 to T=1 collecting magnetization stats
-- to compare against previous runs
for T=100,0,-1 do
	print("Temperature="..T)
	af:set({0,0,0})
	th:setTemperature(T)
	run(0.7)
	resetMagStats()
	run(0.3, collectMagStats, 0.01)
	
	local stats = calculateMagStats()
	
	-- prefixing mag data with temp, time and field and 
	f:write(th:temperature() .. "\t")
	f:write(ss:time() .. "\t")
	f:write(af:get()[3] .. "\t")
	reportMagStats(stats, cols, f)
end
-- end
f:close()


function doLoop(T)
	fn = string.format("MH_%05.1fT_%02iL_%05.3fR.dat", T, L, rate)
	print(fn)
	g = io.open(fn, "w")
	g:write("#" .. table.concat(header, "\t") .. "\n")
	local hmin, hmax, step = -400, 400, 8

	th:setTemperature(T)

	local function field_val(g, H, step)
		af:set(0,0,H)
		local time_per_field = 1.0 / rate
		local dt = time_per_field * step
		print(dt,H)
		
		run(0.7 * dt)
		resetMagStats()

		run(0.3 * dt, collectMagStats, 0.03 * dt)

		local stats = calculateMagStats()

		-- prefixing mag data with temp, time and field and 
		g:write(th:temperature() .. "\t")
		g:write(ss:time() .. "\t")
		g:write(af:get()[3] .. "\t")
		reportMagStats(stats, cols, g)
		g:flush()
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



error()

ss:setTime(-2) -- give the system some time to reach equilibrium
-- core/surf net mag and core num samples

f = io.open("mt" .. L .. ".dat", "w")
f:write("# SphericalParticle2.lua script results\n")
f:write("# temp cm sm tm\n")

-- save configuration to file
function report()
-- 	t = string.format("%05.3f", ss:time()) --formatted time
-- 	g = io.open("state-" .. t .. ".dat", "w")
-- 	for z=1,dims[3] do
-- 		for y=1,dims[2] do
-- 			for x=1,dims[1] do
-- 				sx, sy, sz = ss:spin(x,y,z)
-- 				if sx^2 + sy^2 + sz^2 > 1e-4 then
-- 					g:write(table.concat({x, y, z, sx, sy, sz}, "\t") .. "\n")
-- 				end
-- 			end
-- 		end
-- 	end
-- 	g:close()
-- 	print(t)
end

function avrgMag(region)
	local avrg = 0
	local count = 0
	for k,v in pairs(region) do
		avrg = avrg + v
		count = count + 1
	end
	return avrg / count
end


-- Simulation main loop
while current_temp >= 0 do
	local t = ss:time()
	print(t, next_step, current_temp)
	-- check for next temperature step
	if t >= next_step then
		if current_temp < 6 then
			temp_step = last_temp_steps
		end
		
		for k,v in pairs(mags) do
			
		end
		
		if cn > 0 then
			w = {current_temp, cm / cn, sm / sn, (cm+sm) / (cn+sn) }
			line = table.concat(w, "\t")
			f:write(line .. "\n")
			f:flush() -- so we don't have to wait to see the data
			print(line)
			report()
		end

		current_temp = current_temp + temp_step
		th:setTemperature(current_temp)
		-- reset core/surf net mag and core num samples
		cm, cn = 0, 0
		sm, sn = 0, 0
		
		-- let the system adjust to the new temp
		next_sample= t + eq_time
		next_step  = t + time_per_temp
	end

	step() -- do 1 LLG step
	
	-- check for sampling step
	if t >= next_sample then
		for k1,region in pairs(regions) do
			for k2,site in pairs(region) do
				local sx, sy, sz = ss:spin(site.x, site.y, site.z)
				table.insert(mags[k1], (sx^2 + sy^2 + sz^2)^(1/2))
			end
		end
		next_sample = ss:time() + sample_dt
	end
end
f:close()
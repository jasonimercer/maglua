-- This is a template showing how to implement a core for the KMC framework

ss = SpinSystem.new(32,32,2)

ex = Exchange.new(ss)
ani = Anisotropy.new(ss)
mag = Magnetostatics2D.new(ss)
thermal = Thermal.new(ss, Random.Isaac.new())
zee = AppliedField.new(ss)

kmc = KMC.new() -- empty framework, needs some functions and info

kmc:setFields({ex,ani,mag,thermal,zee})

-- this is the function that computes the time and event for a site
function time_event(kmc, ss, pos)
	local fields_rng, fields_det = kmc:fields("Thermal")

	local time = magic_physics(ss, pos, fields_det)

	-- define what the event is via a function that acts on a SpinSystem
	local function event(ss)
		local mx,my,mz = ss:spin(pos)
		ss:setSpin(pos, {mx,my,-mz})
	end

	return time, event
end

kmc:setEventFunction(time_event)

thermal:setTemperature(5)

-- We could have used kmc:setTemperature(5)

-- main loop:
while ss:time() < 10 do
	kmc:apply(ss) -- this call is internally parallelized via MPI
end


-- Section: Basic Micromagnetic Simulation
--
-- In this section, we will script and run a simple micromagnetic simulation.
-- First we will annotate our code and define our simulation parameters:

 -- This is a basic micromagnetic simulation
   nx, ny = 16, 16 -- Dimensions of the simulation
        J = 1      -- Exchange Strength
        g = 0.1    -- Dipolar strength
       dt = 0.01   -- Time Step
report_dt = 1.0    -- Reporting time step
     maxt = 30     -- Max Time
    alpha = 0.5    -- Damping

-- It is customary (but not required) to define these values at the top of the 
-- script so that they may be changed easily if needed. Now we will create our 
-- Data, Field and LLG objects based on these values.

ss  = SpinSystem.new(nx, ny)
ex  =   Exchange.new(nx, ny)
dip =     Dipole.new(nx, ny)
llg =        LLG.new("Quaternion")

print(ss, ex, dip, llg)

-- Next we assign our simulation parameters to the objects. Remember, since these 
-- are objects, we need to use the ":" operator to access the methods

dip:setStrength(g)
 ss:setAlpha(alpha)
 ss:setTimeStep(dt)

-- The exchange strength, J, has not been set yet. This is done when we setup
-- "Exchange Pathways" between sites, which we will do while setting up
-- a random inplane initial orientation

for j=1,ny do
	for i=1,nx do
		theta = math.random() * 2 * math.pi
		ss:setSpin({i,j}, {math.cos(theta), math.sin(theta), 0})

		-- PBC are automatic
		ex:addPath({i,j}, {i+1, j}, J)
		ex:addPath({i,j}, {i-1, j}, J)
		ex:addPath({i,j}, {i, j+1}, J)
		ex:addPath({i,j}, {i, j-1}, J)
	end
end

-- Our simulation is now initialized. Lets calculate the starting average
-- magnetization vector. We will put this in a function so we can use it later.

function print_netmag(prefix_text)
	mx, my, mz = 0, 0, 0

	for j=1,ny do
		for i=1,nx do
			sx, sy, sz = ss:spin(i,j)

			mx = mx + sx
			my = my + sy
			mz = mz + sz
		end
	end

	mx = mx / (nx * ny)
	my = my / (nx * ny)
	mz = mz / (nx * ny)

	print(prefix_text .. math.sqrt(mx^2 + my^2 + mz^2))
end

print_netmag("|M| = ")

-- Alternatively, we could have constructed this function using the `netMag'
-- method of the SpinSystem but this is more instructive.
--
-- Lets create a function to evaluate 1 simulation step. We will calculate the
-- fields and make 1 LLG step.

function step()
	ss:zeroFields() --clear fields from last step

	ex:apply(ss)    -- calculate exchange fields of the SpinSystem
	dip:apply(ss)   -- dipolar fields

	ss:sumFields()  -- add the fields together to get an effective field

	llg:apply(ss)   -- make 1 LLG step
end

-- Now that we have this function, we can easily write a main simulation loop
-- and print the net magnetization every report_dt.

next_report = 0
while ss:time() < maxt do
	step()

	if ss:time() >= next_report then
		print_netmag(ss:time() .. "\t")
		next_report = next_report + report_dt
	end
end

-- 
-- This example wraps an RK4 integration step in an adaptive timestep scheme
-- A system evolves to a final configuration and is written as a POVRay file
-- 

dofile("maglua://RungeKutta.lua")
dofile("maglua://AdaptiveTimeStep.lua")
dofile("maglua://POVRay.lua")

ss   = SpinSystem.new(30,30)
ex   = Exchange.new(ss)
dip  = Dipole.new(ss)
ani  = Anisotropy.new(ss)
temp = Thermal.new(ss, Random.Isaac.new())
llg  = LLG.Quaternion.new()

--1D interpolation object used as {time, temperature}
temperature = Interpolate.new({{0,10}, {50,0.5}, {150,0.5}, {190, 0.1}, {195,0},{1e8,0}}) 

max_time = 200

ss:setAlpha(0.1)
ex_str = 1/2
ani_str = 5/2

for i=1,ss:nx() do
	for j=1,ss:ny() do
		ss:setSpin({i,j}, {1,0,0}, 1)

 		ex:add({i,j}, {i+1,j}, ex_str)
		ex:add({i,j}, {i-1,j}, ex_str)
		ex:add({i,j}, {i,j+1}, ex_str)
		ex:add({i,j}, {i,j-1}, ex_str)
			
		ani:add({i,j}, {0,0,1}, ani_str)
	end
end

for i=1,ss:nx() do
	ss:setSpin({ i, 1}, {0,0,0})
	ss:setSpin({ i,11}, {0,0,0})
end
for j=1,ss:ny() do
	ss:setSpin({ 1, j}, {0,0,0})
	ss:setSpin({11, j}, {0,0,0})
end

function calcField(ss)
	ss:resetFields()
	ex:apply(ss)
	dip:apply(ss)
	ani:apply(ss)
	ss:sumFields()
end

function dynamics(ss)
	temp:set(temperature:value(ss:time()))
end

local tol = 5e-4 * 30^2
stepRK4 = make_rk_step_function(ss, "RK4", calcField, llg, temp)
step    = make_adapt_step_function(ss, stepRK4, tol, dynamics, llg, temp)

while ss:time() < max_time do
	step(ss)
end

POVRay("SpinSystem.pov", ss)

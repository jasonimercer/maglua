-- This script simulates magnetostatic interaction between 2 layers with differing grain
-- heights. The bottom layers (layer 1) has grain heights of 10 nm. There is a 6 nm gap
-- between the layers and the top layer (2) is 5 nm tall. Each layer has grains that are 
-- 10 nm by 10 nm and the layers are offset in the XY plane by 5 nm X and 5 nm Y.
-- 

dofile("maglua://CGS.lua")
dofile("maglua://RungeKutta.lua")

ss  = SpinSystem.new(16,16,2)
mag = Magnetostatics2D.new(ss)
zee = AppliedField.new(ss)
llg = LLG.Cartesian.new(ss)
temp= Thermal.new(ss, Random.Isaac.new())
int = Interpolate.new({{0,1400}, {30*ns,0}, {100*ns,0}})

A, B, C = {10*nm, 0, 0}, {0, 10*nm, 0}, {0*nm,0*nm,10*nm}

mag:setUnitCell(1, A, B, C)
mag:setUnitCell(2, A, B, C) --this one doesn't matter

mag:setGrainSize(1, 10*nm, 10*nm, 10*nm)
mag:setGrainSize(2, 10*nm, 10*nm, 10*nm)

cell1 = 10*10*10*nm^3
cell2 = 10*10*10*nm^3

temp:set(300*kB*Kelvin)


zee:set({0*Oe, 0, 0})

Ms = 1400 * emu/cc
local function r()
	local function r1()
		return math.random()*2-1
	end
	return {r1(), r1(), r1()}
end

for x=1,16 do
	for y=1,16 do
		ss:setSpin({x,y,1}, r(), Ms*cell1)
		ss:setSpin({x,y,2}, r(), Ms*cell2)
	end
end

ss:setTimeStep(5*ps)
ss:setGamma(gamma) --from CGS file
ss:setAlpha(1.0)

function calcFields(s)
-- 	temp:set(int(s:time()) * kB * Kelvin)
	s:resetFields()
	mag:apply(s)
	zee:apply(s)
	s:sumFields()
end

stepRK4 = make_rk_step_function(ss, "RK4", calcFields, llg, temp)

function report()
	local mx, my, mz, mm = ss:netMoment(1/(Ms * (16^2* (cell1 + cell2))))
	print(mx,my,mz,mm)
end

report()
-- while ss:time() < 250*ns do
	stepRK4()
	print(ss:time())
-- 	report()
-- end
report()

-- dofile("maglua://POVRay.lua")
-- POVRay("render.pov", ss, {scale=(1/(Ms*cell1))})

dofile("mag2d_luafuncs_file.lua")

mag2d_save(mag, "save_test.lua")

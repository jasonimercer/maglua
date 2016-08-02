-- This script simulates magnetostatic interaction between 2 layers with differing grain
-- heights. The bottom layers (layer 1) has grain heights of 10 nm. There is a 6 nm gap
-- between the layers and the top layer (2) is 5 nm tall. Each layer has grains that are 
-- 10 nm by 10 nm and the layers are offset in the XY plane by 5 nm X and 5 nm Y.
-- 

dofile("maglua://CGS.lua")
dofile("maglua://RungeKutta.lua")

ss  = SpinSystem.new(5,5,5)
mag = Magnetostatics3D.new(ss)
llg = LLG.Cartesian.new(ss)
A, B, C = {10*nm, 0, 0}, {0, 10*nm, 0}, {0*nm,0*nm,10*nm}

mag:setUnitCell(A, B, C)
mag:setGrainSize(10*nm, 10*nm, 10*nm)

mag:setTruncation(2, 2, 2, 2)

cell = 10*10*10*nm^3

Ms = 1400 * emu/cc



for x=1,ss:nx() do
	for y=1,ss:ny() do
		for z=1,ss:nz() do
			ss:setSpin({x,y,z}, {1,0,0}, Ms*cell)
		end
	end
end

ss:setTimeStep(5*ps)
ss:setGamma(gamma) --from CGS file
ss:setAlpha(1.0)

function calcFields(s)
-- 	temp:set(int(s:time()) * kB * Kelvin)
	s:resetFields()
	mag:apply(s)
	s:sumFields()
end

stepRK4 = make_rk_step_function(ss, "RK4", calcFields, llg, temp)

function report()
	local mx, my, mz, mm = ss:netMoment(1/(Ms * (ss:nx()*ss:ny()*ss:nz()*cell)))
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

-- dofile("mag2d_luafuncs_file.lua")

-- mag2d_save(mag, "save_test.lua")

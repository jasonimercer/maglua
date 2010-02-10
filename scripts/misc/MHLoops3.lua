-- Script to attempt to reproduce Martin's Sheinfein MHLoops

eV, Tesla, ps, nm, Kelvin = 1, 1, 1, 1, 1

n        = 10
uB       = 5.788321e-5 * eV / Tesla
Oe       = 1e-4 * Tesla
Gauss    = Oe
erg      = 6.24150974e11 * eV
emu      = erg/Gauss
cm       = 1e7 * nm
cc       = cm^3
a        = (10 * nm)
cell     = a^3
hbar     = 6.58211e-4 * eV * ps
kB       = 8.617343e-5 * eV / Kelvin

Field    = 2000 * Oe
Ms       =  400 * emu/cc * cell
Ku       =  4e5 * erg/cc
A        = 0.05 * 10^-6 * erg/cm 
Temp     = 500 * Kelvin
alpha    = 2.0
dt       = 0.1 * ps / hbar
filename = "mhA0.05_T500.dat"

ss      = SpinSystem.new(n, n, n)
field   = AppliedField.new(n, n, n)
ani     = Anisotropy.new(n, n, n)
ex      = Exchange.new(n, n, n)
thermal = Thermal.new(n, n, n)
rng     = MTRand.new()
rng:setSeed(493188)

thermal:setTemperature(Temp * kB / 2)


llg = LLGCartesian.new()
-- llg = LLGQuaternion.new()

print("Filename       " .. filename)
print("Ms per cell is " .. Ms .. " eV/T")
print("Ms per cell is " .. Ms/uB .. " nB")
print("LLG Type       " .. llg:type() )
print("Anisotropy is  " .. 4e5 .. " erg/cc")
print("Anisotropy is  " .. Ku*cell .. " eV")


llg:setAlpha(alpha)
llg:setTimeStep(dt)
llg:setGamma(2.0) -- gotta figure out why gamma = 2
field:set(0,0,Field*uB)

-- exchange neighbour directions
dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}

-- setup spins
for k=1,n do
	for j=1,n do
		for i=1,n do
			ss:setSpin(i, j, k, 0, 0, -Ms/uB)

			x, y, z = rng:normal(), rng:normal(), rng:normal()
			ixyz = 1.0/math.sqrt(x*x + y*y + z*z)
			x, y, z = x*ixyz, y*ixyz, z*ixyz
			ani:setSite(i, j, k, x, y, z, 2*Ku*cell)

			-- non-pbc J
			for _,d in pairs(dir) do
				dx, dy, dz = i+d[1], j+d[2], k+d[3]
				if ex:member(dx, dy, dz) then
					ex:addPath(i, j, k, dx, dy, dz, 1.5*2*A*a*uB*uB/10)
				end
			end
		end
	end
end



f = io.open(filename, "w")
f:write("# time(ps)	h(Oe)	sx	sy	sz	|s|\n")

-- putting all the fields into the list "h"
h = {}
for i=-5,5 do
	table.insert(h, i/25 * 2000)
end
for i=4,-5,-1 do
	table.insert(h, i/25 * 2000)
end

inv = 1.0 / (Ms/uB * n^3)
for i,H in pairs(h) do

	Field = H
	field:set(0,0,Field*Oe*uB)

-- 	s = {}
-- 	n = 0
-- 	slope = 0

-- 	while n < 2 or math.abs(slope) > 1E-4 do
		current = ss:time()*hbar
		while ss:time()*hbar < current + 10000 do
			ss:zeroFields()
	
			field:apply(ss)
			ani:apply(ss)
			ex:apply(ss)
			thermal:apply(llg, rng, ss)

			ss:sumFields()
			llg:apply(ss)
		end

		time, x, y, z, m = ss:time()*hbar, ss:netMag()
		line = table.concat({time, Field,x*inv,y*inv,z*inv,m*inv}, "\t")

-- 		table.insert(s, 1, {time, m*inv})

-- 		n = n + 1
-- 		if n >= 2 then
-- 			slope = (s[2][2] - s[1][2])/(s[2][1] - s[1][1])
-- 		end

-- 		print(line, slope)
-- 	end

	f:write(line .. "\n")
	f:flush()
end

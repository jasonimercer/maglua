-- Script to attempt to reproduce Martin's Sheinfein MHLoops

eV, Tesla, ps, nm = 1, 1, 1, 1

n        = 10
uB       = 5.788321e-5 * eV / Tesla
Oe       = 1e-4 * Tesla
Gauss    = Oe
erg      = 6.24150974e11 * eV
emu      = erg/Gauss
cm       = 1e7 * nm
cc       = cm^3
cell     = (10 * nm)^3
hbar     = 6.58211e-4 * eV * ps

Field    = 2000 * Oe
Ms       =  400 * emu/cc * cell
K        =  4e5 * erg/cc 
alpha    = 0.5
dt       = 0.1 * ps / hbar
filename = "output.dat"

ss      = SpinSystem.new(n, n, n)
field   = AppliedField.new(n, n, n)
ani     = Anisotropy.new(n, n, n)
rng     = MTRand.new()
rng:setSeed(12345)

llg = LLGCartesian.new()

print("Filename       " .. filename)
print("Ms per cell is " .. Ms .. " eV/T")
print("Ms per cell is " .. Ms/uB .. " nB")
print("LLG Type       " .. llg:type() )
print("Anisotropy is  " .. 4e5 .. " erg/cc")
print("Anisotropy is  " .. K*cell .. " eV")


llg:setAlpha(alpha)
llg:setTimeStep(dt)
llg:setGamma(2.0) -- gotta figure out why gamma = 2
field:set(0,0,Field*uB)

-- setup spins
for k=1,n do
	for j=1,n do
		for i=1,n do
			ss:setSpin(i, j, k, Ms/uB, 0, 0)

			x, y, z = rng:normal(), rng:normal(), rng:normal()
			ixyz = 1.0/math.sqrt(x*x + y*y + z*z)
			x, y, z = x*ixyz, y*ixyz, z*ixyz
			ani:setSite(i, j, k, x, y, z, 2*K*cell)
		end
	end
end



f = io.open(filename, "w")
f:write("# time(ps)	h(Oe)	sx	sy	sz	|s|\n")

-- putting all the fields into the list "h"
h = {}
for i=-50,50 do
	table.insert(h, i/50 * 2000)
end
for i=50,-50,-1 do
	table.insert(h, i/50 * 2000)
end
	



inv = 1.0 / (Ms/uB * n^3)
for i,H in pairs(h) do

	Field = H
	field:set(0,0,Field*Oe*uB)

	s = {}
	n = 0
	slope = 0

	while n < 2 or math.abs(slope) > 0.000005 do
		current = ss:time()*hbar
		while ss:time()*hbar < current + 20 do
			ss:zeroFields()
	
			field:apply(ss)
			ani:apply(ss)
			ss:sumFields()
	
			llg:apply(ss)
		end

		time, x, y, z, m = ss:time()*hbar, ss:netMag()
		line = table.concat({time, Field,x*inv,y*inv,z*inv,m*inv}, "\t")

		table.insert(s, 1, {time, m*inv})

		n = n + 1
		if n >= 2 then
			slope = (s[2][2] - s[1][2])/(s[2][1] - s[1][1])
		end

		print(line, slope)
	end

	f:write(line .. "\n")
	f:flush()
end

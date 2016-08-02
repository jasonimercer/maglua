-- Script to attempt to reproduce Martin's Sheinfein MH loops
--
-- Random initial configuration
-- Random Anisotropy angles
--
-- Ms = 400 emu/cc
-- Ku = 4x10^5 erg/cc
-- A = {0, 0.1, 0.05} x10^-6 erg/cm
-- M is initially along x
-- H = -2000 Oe to 2000 Oe
-- Alpha = 0.5
-- no dipole
-- command line arguments are:
--
--  argv[3] = "q" or "c" for quaternion or cartesian LLG
--  argv[4] = exchange x10^-6 erg/cm
--  argv[5] = timestep in picoseconds
--  argv[6] = time for each field in picoseconds
--  argv[7] = output filename
--
-- Example 
-- ./maglua  loops.lua c 0.0 0.1 2.0 output.dat
-- ./maglua  loops.lua   c/q   A   dt   t/h   output.dat

eV, Tesla, ps, nm = 1, 1, 1, 1

n        = 10
uB       = 5.788321e-5 * eV / Tesla
Oe       = 10^-4 * Tesla
Gauss    = Oe
erg      = 6.24150974e11 * eV
emu      = erg/Gauss
cm       = 1e7 * nm
cc       = cm^3
cell     = (10 * nm)^3
hbar     = 6.58211e-4 * eV * ps

-- Field    = 2000 * Oe
Ms       = 400 * emu/cc * cell
K        = 4*10^5 * erg/cc
A        = tonumber(argv[4]) * erg/cm
alpha    = 0.5
dt       = tonumber(argv[5]) * ps / hbar
maxtau   = 734 * ps / hbar
filename = argv[7]

-- report_dt   = tonumber(argv[6]) * ps / hbar
-- next_report = report_dt

ss      = SpinSystem.new(n, n, n)
field   = AppliedField.new(n, n, n)
ani     = Anisotropy.new(n, n, n)
ex      = Exchange.new(n, n, n)
rng     = MTRand.new()

if argv[3] == "c" then llg = LLGCartesian.new()  end
if argv[3] == "q" then llg = LLGQuaternion.new() end

print("Filename       " .. filename)
-- print("Field is       " .. Field .. " Tesla")
-- print("Field is       " .. Field*uB .. " eV")
print("Ms per cell is " .. Ms .. " eV/T")
print("Ms per cell is " .. Ms/uB .. " nB")
print("LLG Type       " .. llg:type() )
print("Anisotropy is  " .. 4e5 .. " erg/cc")
print("Anisotropy is  " .. K*cell .. " eV")
print("Exchange is    " .. 1e-6 .. " erg/cm")
print("Exchange is    " .. A .. " eV/nm")
print("Exchange is    " .. A * (10 * nm) .. " eV (dist = 10nm)")


llg:setAlpha(alpha)
llg:setTimeStep(dt)
llg:setGamma(2.0) -- gotta figure out why gamma = 2
-- field:set(0,0,Field*uB)

-- setup spins
dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}
for k=1,n do
	for j=1,n do
		for i=1,n do
			x, y, z = rng:normal(), rng:normal(), rng:normal()
			isl = (Ms/uB)/math.sqrt(x*x + y*y + z*z)
			x, y, z = x*isl, y*isl, z*isl
			ss:setSpin(i, j, k, x, y, z)

			x, y, z = rng:normal(), rng:normal(), rng:normal()
			ixyz = math.sqrt(x*x + y*y + z*z)
			x, y, z = x*ixyz, y*ixyz, z*ixyz
			ani:setSite(i, j, k, x, y, z, 2*K*cell)

			-- non-pbc J
			for _,d in pairs(dir) do
				dx, dy, dz = i+d[1], j+d[2], k+d[3]
				if ex:member(dx, dy, dz) then
					ex:addPath(i, j, k, dx, dy, dz, (A * (10 * nm)) * uB^2)
				end
			end
		end
	end
end


f = io.open(filename, "w")
f:write("# Attempt to reproduce Martin's Sheinfein MH loops\n")
f:write("#\n")
f:write("# Random initial configuration\n")
f:write("# Random Anisotropy angles\n")
f:write("#\n")
f:write("# Ms = 400 emu/cc\n")
f:write("# Ku = 4x10^5 erg/cc\n")
f:write("# A = " .. argv[4] .. "x10^6 erg/cc\n")
f:write("# M is initially along x\n")
f:write("# H = -2000 Oe to 2000 Oe\n")
f:write("# Alpha = " .. alpha .. "\n")
f:write("# no dipole\n")
f:write("# ===============================================================\n")
f:write("# time(ps)	H(Oe)	sx	sy	sz	|s|\n")



-- will run the sim at these fields
H = {}
for i=25,-25,-1 do
	table.insert(H, i*80)
-- 	table.insert(H, -2000)
end
for i=-24,25 do
	table.insert(H, i*80)
-- 	table.insert(H, -2000)
end


i = Ms/uB * n^3

for _,h in pairs(H) do
	current = ss:time()
	nexttime = current + tonumber(argv[6]) * ps / hbar
	field:set(0,0,h*uB)
-- 	print("Running at " .. h .. " Oe")

	while ss:time() < nexttime do
		ss:zeroFields()
	
-- 		ex:apply(ss)
		field:apply(ss)
		ani:apply(ss)
	
		ss:sumFields()
		llg:apply(ss)
	end

	time, x, y, z, m = ss:time()*hbar, ss:netMag()
	d = table.concat({time,h,x/i,y/i,z/i,m/i}, "\t")
	print(d)
	f:write(d .. "\n")
end

f:close()

-- Script to attempt to reproduce Martin's Sheinfein test2
--
-- Ms = 400 emu/cc
-- K = 4x10^5 erg/cc
-- A = 1x10^-6 erg/cm
-- M is initially along x
-- H = 2000 Oe
-- Alpha = 0.1
-- No exchange, no anisotropy, no dipole
-- command line arguments are:
--
--  argv[3] = "q" or "c" for quaternion or cartesian LLG
--  argv[4] = timestep in picoseconds
--  argv[5] = reportstep in picoseconds
--  argv[6] = output filename
--
-- Example 
-- ./maglua  MSTest3.lua c 0.1 2.0 cart0.10.dat

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
A        = 1e-6 * erg/cm
alpha    = 0.1
dt       = tonumber(argv[4]) * ps / hbar
maxtau   = 734 * ps / hbar
filename = argv[6]

report_dt   = tonumber(argv[5]) * ps / hbar
next_report = report_dt

ss      = SpinSystem.new(n, n, n)
field   = AppliedField.new(n, n, n)
ani     = Anisotropy.new(n, n, n)
ex      = Exchange.new(n, n, n)

if argv[3] == "c" then llg = LLGCartesian.new()  end
if argv[3] == "q" then llg = LLGQuaternion.new() end

print("Filename       " .. filename)
print("Field is       " .. Field .. " Tesla")
print("Field is       " .. Field*uB .. " eV")
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
field:set(0,0,Field*uB)

-- setup spins
dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}
for k=1,n do
	for j=1,n do
		for i=1,n do
			ss:setSpin(i, j, k, Ms/uB, 0, 0)
			ani:setSite(i, j, k, 0, 0, 1, 2*K*cell)

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



-- function to turn a list of args into a tab
-- delimited string
function tab(...)
	local s = ""
	for i=1,arg.n-1 do
		s = s .. arg[i] .. "\t"
	end
	return s .. arg[arg.n] .. "\n"
end

f = io.open(filename, "w")
f:write("# Time test dataset. Attempting to reproduce Martin's results for\n")
f:write("# Ms = 400 emu/cc\n")
f:write("#  K = 4x10^5 erg/cc\n")
f:write("# M is initially along x\n")
f:write("# H = 2000 Oe is along z\n")
f:write("# Alpha = 0.1\n")
f:write("# No exchange, no anisotropy, no dipole\n")
f:write("# ===============================================================\n")
f:write("# time(ps)	sx	sy	sz	|s|\n")

i = Ms/uB * n^3
while ss:time() < maxtau do
	ss:zeroFields()

	ex:apply(ss)
	field:apply(ss)
	ani:apply(ss)

	ss:sumFields()
	llg:apply(ss)

	tau, x, y, z, m = ss:time(), ss:netMag()
	if tau >= next_report - dt/2 then
		next_report = next_report + report_dt
		f:write(tab(tau*hbar,x/i,y/i,z/i,m/i))
	end
end
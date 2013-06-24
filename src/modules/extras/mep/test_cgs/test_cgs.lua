-- 
-- This is an MEP example
-- 
-- 2 SW particles
-- Low exchange coupling 
-- Field with perpendicular component
-- Initially unknown minima (but guesses are known)
-- 
-- This script will:
-- Setup a problem
-- Refine initial MEP end points
-- Run the adaptive step MEP algorithm
-- Refine the minima and saddle points
-- Plot energy vs path for initial and final paths (MEP.png)
-- Plot energy vs reduced configuration for initial and final paths (MAP.png)
-- Calculate curvature related info for minima and saddle points
-- Print initial and final energy barriers
-- 

dofile("maglua://CGS.lua") -- units
dofile("landscape_plot.lua") -- nice custom plotting routine

nx,ny,nz = 1,1,2

ss = SpinSystem.new(nx,ny,nz)
zee = AppliedField.new(ss)
ani = Anisotropy.new(ss)
ex = Exchange.new(ss)
 
Ms =   500 *  emu/cc
 I =   0.2 * erg/cm/cm
 K = {1.5e6 * erg/cc, 3.0e6*erg/cc} --different K values for each layer

 
 H = {7500*Oe, 0*Oe, -1000*Oe} -- applied field

a = 6*nm
cell = a^3

-- exchange field
Hex_z  = I/(a*Ms)

ex:setPeriodicXYZ(true, true, false)
for p,m in ss:eachSite() do
	local i,j,k = p[1], p[2], p[3]
	
	ss:setSpin(p, {0,0,1}, Ms*cell)
	
	ani:add(p, {0,0,1}, K[k]*cell )

	ex:add({i,j,k-1}, p, Hex_z)
	ex:add({i,j,k+1}, p, Hex_z)
end

zee:set(H)

function energy(ss)
	  ss:resetFields()
 	  ex:apply(ss)
	 ani:apply(ss)
	 zee:apply(ss)
	  ss:sumFields()

	local E = -(ss:spinArrayX():dot(ss:fieldArrayX("Total"))+ 
			    ss:spinArrayY():dot(ss:fieldArrayY("Total"))+
			    ss:spinArrayZ():dot(ss:fieldArrayZ("Total")))
	E = E/2.0
	return E 
end

-- support function for recording information
function writeEnergyPath(filename, mep, offset)
	offset = offset or 0
	f = io.open(filename, "w")
	local ee = mep:pathEnergy()
	for i=1,table.maxn(ee) do
		f:write(i + offset .. "\t" .. ee[i] .. "\n")
	end
	f:close()
end

-- support function for recording information
function writePathComponents(filename, mep)
	f = io.open(filename, "w")
	
	local np = mep:numberOfPathPoints()
	local ns = mep:numberOfSites()
	local ee = mep:pathEnergy()

	for p=1,np do
		local line = {}
		for s=1,ns do
			local x, y, z = mep:spin(p,s)
			local r = (x^2 + y^2 + z^2)^(1/2)
			table.insert(line, math.acos(z/r))
		end
		table.insert(line, ee[p])
		line = table.concat(line, "\t")
		f:write(line .. "\n")
	end
	
	f:close()
end


-- These are the sites involved in the
-- MEP calculation. Order has meaning here.
sites = {{1,1,1}, {1,1,2}}

local M = Ms*cell
upup     = {{0,0, M}, {0,0, M}}
downdown = {{0,0,-M}, {0,0,-M}}
initial_path = {upup, downdown}

np = 8

mep = MEP.new()
mep:setSites(sites)
mep:setInitialPath(initial_path)
mep:setEnergyFunction(energy)
mep:setNumberOfPathPoints(np)
mep:setSpinSystem(ss)
mep:setTolerance(0.1)


-- These few lines may look a bit strange.
-- We're initializing to 2 points and then relaxing them.
-- This ensures we're flipping from one minimum to another
-- provided the initial_path is near minima
mep:initialize(2)
mep:relaxSinglePoint(1, 100)
mep:relaxSinglePoint(2, 100)

-- initialize to real number of points for the calcualtion
mep:initialize(np) 


local initial_energy_barrier = mep:energyBarrier()



writePathComponents("InitialComponents.dat", mep)
writeEnergyPath("InitialEnergyPath.dat", mep)

mep:compute(20) -- Here is the MEP calculation 

-- finding maximal points and refining them individually
mins, maxs, all = mep:maximalPoints()

for k,v in pairs(maxs) do
	mep:relaxSaddlePoint(maxs[k], 25)
end

for k,v in pairs(mins) do
	mep:relaxSinglePoint(mins[k], 25)
end

writePathComponents("FinalComponents.dat", mep)
writeEnergyPath("FinalEnergyPath.dat", mep)	

-- Now we're into the plotting parts
local ee = mep:pathEnergy()

local circles = {}
for i,idx in pairs(all) do
	local x = idx
	local y = ee[x]
	table.insert(circles, string.format("set object %d circle at %g,%g size 0.1 fc rgb \"navy\"\n", i, x,y))
end

cmd = 
table.concat(circles, "") ..
[[
	set xlabel "Energy Path"
	set ylabel "Energy (ergs)"
    set term svg size 800,600 enhanced fname "Times" fsize 14 lw 1
    set output "plot1.svg"
	set key bottom left
	set title "Initial and Final MEPs. Important sites marked."
	plot "InitialEnergyPath.dat" with lines title "Initial Path", "FinalEnergyPath.dat" w l t "Final Path"
]]

f = io.open("plot1.cmd", "w")
f:write(cmd)
f:close()

os.execute("gnuplot plot1.cmd && convert plot1.svg MEP.png && rm plot1.cmd plot1.svg")


local files = {"InitialComponents.dat", "FinalComponents.dat"}

-- nice looking landscape plot (in another file)
landscape_plot(files)




-- From here on it's command line info
-- This function prints an array in a matrix form
function printMat(name, M)
	local info = {string.format("% 3s", name), string.format("(%dx%d)", M:ny(), M:nx())}
	for r=1,M:ny() do
		local t = {info[r] or ""}
		for c=1,M:nx() do
  			--table.insert(t, string.format("% 06.6f", M:get(c,r)))
			table.insert(t, string.format("% 06.6e", M:get(c,r)))
		end
		print(table.concat(t, "\t"))
	end
	print()
end

-- prints all the curvature related info for a given point
function point_data(idx, _mep)
	_mep = _mep or mep
	local ns = _mep:numberOfSites()
	local line = {}
	for s=1,ns do
		local x, y, z = _mep:spin(idx, s)
		table.insert(line, string.format("{%g, %g, %g}", x,y,z))
	end
	line = table.concat(line, ", ")
	print("configuration at point " .. idx)
	print(line)

	D = _mep:hessianAtPoint(idx)
	print("Hessian at point " .. idx)
	printMat("D", D)

	vals, vecs = D:matEigen()

	print("Eigen Values at point " .. idx)
	printMat("evals", vals)

	print("Eigen Vectors at point " .. idx .. " (rows)")
	printMat("evecs", vecs)

	print()
end

print("Data for maximal points:")
for k,pidx in pairs(all) do
	point_data(pidx)
end


local final_energy_barrier = mep:energyBarrier()

print("Initial Energy Barrier (erg): ", initial_energy_barrier)
print("  Final Energy Barrier (erg): ", final_energy_barrier)



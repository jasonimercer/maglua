ss = SpinSystem.new(4,4,2)
zee = AppliedField.new(ss)
ani = Anisotropy.new(ss)
 ex = Exchange.new(ss)

ex:setPeriodicXYZ(true, true, false)
 
K, Jxy, Jz = 2, 0.0, 0.8
for k=1,ss:nz() do
	for j=1,ss:ny() do
		for i=1,ss:nx() do
			if k == ss:nz() then
				ss:setSpin({i,j,k}, {0,0,1}, 1)
			else
				ss:setSpin({i,j,k}, {0,0,1}, 1)
			end
			
			ani:add({i,j,k}, {0,0,1}, K)
		
			ex:add({i,j,k}, {i+1,j,k}, Jxy)
			ex:add({i,j,k}, {i-1,j,k}, Jxy)
			ex:add({i,j,k}, {i,j+1,k}, Jxy)
			ex:add({i,j,k}, {i,j-1,k}, Jxy)
			ex:add({i,j,k}, {i,j,k-1}, Jz)
			ex:add({i,j,k}, {i,j,k+1}, Jz)
		end
	end
end

zee:set({0,0,-1.0})




function energy(ss)
	ss:resetFields()
	ex:apply(ss)
	ani:apply(ss)
	zee:apply(ss)
	ss:sumFields()

	return -1 * (
		ss:fieldArrayX("Total"):dot(ss:spinArrayX()) +
		ss:fieldArrayY("Total"):dot(ss:spinArrayY()) +
		ss:fieldArrayZ("Total"):dot(ss:spinArrayZ()) )
end


function writeEnergyPath(filename, _mep, _offset)
	_mep = _mep or mep
	_offset = _offset or 0
	f = io.open(filename, "w")
	local ee = _mep:pathEnergy()
	for i=1,table.maxn(ee) do
		f:write(i+_offset .. "\t" .. ee[i] .. "\n")
	end
	f:close()
end

function writePathComponents(filename, _mep)
	_mep = _mep or mep
	f = io.open(filename, "w")
	
	local np = _mep:numberOfPathPoints()
	local ns = _mep:numberOfSites()
	local ee = _mep:pathEnergy()

	for p=1,np do
		local line = {}
		for s=1,ns do
			local x, y, z = _mep:spin(p,s)
			local r = (x^2 + y^2 + z^2)^(1/2)
			table.insert(line, math.acos(z/r))
		end
		table.insert(line, ee[p])
		line = table.concat(line, "\t")
		f:write(line .. "\n")
	end
	
	f:close()
end


function energy_landscape(...)
	local arg = {...}
	local filename = "energy_landscape.txt"
	local ss2 = ss:copy()
	local f = io.open(filename, "w")
	
	local dt=math.pi/128
	for t1=0,math.pi+dt/2,dt do
		local d1 = {math.sin(t1), 0, math.cos(t1)}
		ss2:setSpin({2,2,1}, d1)
		for t2=0,math.pi+dt/2,dt do
			local d2 = {math.sin(t2), 0, math.cos(t2)}
			ss2:setSpin({2,2,2}, d2)
			f:write(table.concat({t1,t2,energy(ss2)}, "\t") .. "\n")
		end
		f:write("\n")
	end
	f:close()
	
	
	extra_plots = [[, "InitialComponents.dat" w l lt 1 lw 1.5 , "FinalComponents.dat" w l lt 3 lw 1.5 ]]
	
		
	cmd = string.format([[
		set term wxt size 1280,960
		set xrange [0:pi]
		set yrange [0:pi]
		set xlabel "Theta 1"
		set ylabel "Theta 2"
		
		set xrange [0:pi]
		set yrange [0:pi]
		set isosample 250, 250
		set table 'test.dat'
		splot "./energy_landscape.txt"
		unset table
		
		
		set contour base
		set cntrparam level incremental -130, 0.33, 100
		#set cntrparam levels discrete -126,-122,-121,-120,-119.5,-119,-118,-114,-110
		unset surface
		set table "cont.dat"
		splot "./energy_landscape.txt"
		unset table
		
		reset
		set xrange [0:pi]
		set yrange [0:pi]
		unset key
		#set palette defined ( 0 0 0 1, 0.5 1 1 1, 1 1 0 0 )
		set palette rgbformulae 33,13,10
		
		l '<./cont.sh cont.dat 0 15 0'

		
		plot 'energy_landscape.txt' with image, '<./cont.sh cont.dat 1 15 0' w l lt -1 lw 1.5 %s

		]], extra_plots)

	f = io.open("plot.cmd", "w")
	f:write(cmd)
	f:close()

	os.execute("gnuplot -persist plot.cmd")
end



-- these are the sites involved in the
-- EB calculation. Order has meaning here.
sites = {{2,2,1}, {2,2,2}}

upup     = {{0,0, 1}, {0,0, 1}}
updown   = {{0,0, 1}, {0,0,-1}}
downdown = {{0,0,-1}, {0,0,-1}}
initial_path = {upup, downdown}
-- initial_path = {upup, updown, downdown}

np = 16

mep = MEP.new()
mep:setSites(sites)
mep:setInitialPath(initial_path)
mep:setEnergyFunction(energy)
mep:setNumberOfPathPoints(np)
mep:setSpinSystem(ss)
mep:setGradientMaxMotion(0.2)


mep:initialize(3)
mep:randomize(0.5)
mep:initialize()

writePathComponents("InitialComponents.dat")
writeEnergyPath("InitialEnergyPath.dat")

mep:compute(25)


mins, maxs, all = mep:maximalPoints()

local stepSize = 0.2
local epsilon = 1e-4
local stepCount = 10

for k,v in pairs(maxs) do
	mep:relaxSaddlePoint(maxs[k], stepSize, epsilon, stepCount)
end

for k,v in pairs(mins) do
	mep:relaxSinglePoint(mins[k], stepSize, epsilon, stepCount)
end

writePathComponents("FinalComponents.dat")
writeEnergyPath("FinalEnergyPath.dat")	
	

energy_landscape()

-- error("A")



function printMat(name, M)
	local info = {string.format("% 3s", name), string.format("(%dx%d)", M:ny(), M:nx())}
	for r=1,M:ny() do
		local t = {info[r] or ""}
		for c=1,M:nx() do
			table.insert(t, string.format("% 06.6f", M:get(c,r)))
		end
		print(table.concat(t, "\t"))
	end
	print()
end

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

	D = _mep:hessianAtPoint(idx, 0.001)
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




-- meps = mep:splitAtPoint(all)
-- 
-- for k,v in pairs(all) do
-- 	print(k,v)
-- end
-- print()
-- for k,v in pairs(meps) do
-- 	print(k, meps[k]:numberOfPathPoints())
-- 	meps[k]:compute(100)
-- 	writeEnergyPath("FinalEnergyPathRelax_" .. k .. ".dat", meps[k], all[k]-1)	
-- end

cmd = [[
	set term wxt size 1024,768
	#plot "./FinalEnergyPath.dat" w lp,  "FinalEnergyPathRelax.dat" w lp, "./InitialEnergyPath.dat" w lp ls 7
	plot "./FinalEnergyPath.dat" w lp,  "./InitialEnergyPath.dat" w lp ls 7
]]

f = io.open("plot.cmd", "w")
f:write(cmd)
f:close()

os.execute("gnuplot -persist plot.cmd")




-- the following will render the states
if false then
ss3 = ss:copy()
	for i=1,np do
		local fn = string.format("ss%04d.pov", i)
		mep:writePathPointTo(i,ss3)
		POVRay(fn, ss3,  {scale=0.5})
		os.execute("povray -D -W640 -H480 " .. fn .. "")
	end
end


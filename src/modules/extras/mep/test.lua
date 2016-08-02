ss = SpinSystem.new(4,4,2)
zee = AppliedField.new(ss)
ani = Anisotropy.new(ss)
 ex = Exchange.new(ss)

ex:setPeriodicXYZ(true, true, false)
 
K, Jxy, Jz = 2, 0.1, 0.9
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

function writeEnergyPath(filename, _mep, _deriv, _offset)
	_mep = _mep or mep
	_offset = _offset or 0
	_deriv = _deriv or 0
	f = io.open(filename, "w")
	local ee, pos, zc = _mep:pathEnergyNDeriv(_deriv)
	
	for i=1,table.maxn(pos) do
		f:write(pos[i] + _offset .. "\t" .. ee[i] .. "\n")
	end
	f:close()
end

function writePathComponents(filename, _mep)
	_mep = _mep or mep
	f = io.open(filename, "w")
	
	local np = _mep:pathPointCount()
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

function getHessian(mep, theta1, theta2)
	-- saving old positions incase we need them later
	local ox1, oy1, oz1, om1 = mep:spin(1,1)
	local ox2, oy2, oz2, om2 = mep:spin(1,2)
	
	mep:setSpin(1, 1, {om1 * math.sin(theta1), 0, om1 * math.cos(theta1)})
	mep:setSpin(1, 2, {om2 * math.sin(theta2), 0, om2 * math.cos(theta2)})
	local hessian = mep:hessianAtPoint(1)

	-- restoring old state
	mep:setSpin(1, 1, ox1, oy1, oz1)
	mep:setSpin(1, 2, ox2, oy2, oz2)
	
	return hessian
end

function illegalDirection(dir, illegal)
	dir:scale( dir:dot(dir)^(-1/2) )
	illegal:scale( illegal:dot(illegal)^(-1/2) )
	
	if math.abs(dir:dot(illegal)) > 0.99 then
		return true
	end
	return false
end

-- need to think about this function a little more. 
function isStableConfiguration(mep, theta1, theta2)
-- 	print("examining ", theta1, theta2)
	local ox1, oy1, oz1, om1 = mep:spin(1,1)
	local ox2, oy2, oz2, om2 = mep:spin(1,2)
	
	local s1 = {om1 * math.sin(theta1), 0, om1 * math.cos(theta1)}
	local s2 = {om2 * math.sin(theta2), 0, om2 * math.cos(theta2)}
	
	mep:setSpin(1, 1, s1)
	mep:setSpin(1, 2, s2)
	
	local hessian = getHessian(mep, theta1, theta2)
	vals, vecs = hessian:matEigen()
	
	local illegal_dir_1 = Array.Double.new(3)
	local illegal_dir_2 = Array.Double.new(3)
	illegal_dir_1:setFromTable(s1)
	illegal_dir_2:setFromTable(s2)
	illegal_dir_1:scale(illegal_dir_1:dot(illegal_dir_1)^(-1/2))
	illegal_dir_2:scale(illegal_dir_2:dot(illegal_dir_2)^(-1/2))
	
	-- this is stable if all eigenvalues are positive
	-- we will allow negative values for illegal directions (grow/shrink spin)
	-- and very small negative values (since the Eigenvalue problem is numerically finicky)

	-- need a scale to help decide if something's zero or not.
	local val_scale = vals:dot(vals)^(1/2) 
	
	local this_is_a_minimum = true --looking to disprove this
	
	-- let's step through the values
	for i=1,vals:nx() do
		if this_is_a_minimum then --try this evalue
			local eigen_value = vals:get(i)
			--print(i, eigen_value)
			
			if math.abs(eigen_value / val_scale) < 0.1 then --it's a zero value
				--print(eigen_value, "is zero")
			else
				if eigen_value < 0 then 
					this_is_a_minimum = false
-- 					-- the only hope is an illegal direction
-- 					local ev_dir1 = vecs:slice({{1,i}, {3,i}})
-- 					local ev_dir2 = vecs:slice({{4,i}, {6,i}})
-- 
-- 										
-- 					if illegalDirection(ev_dir1, illegal_dir_1) or illegalDirection(ev_dir2, illegal_dir_2) then
-- 						--print(eigen_value, "colinear w/ illegal value")
-- 					else
-- 						--print(eigen_value, "is not colinear with illegal value")
-- 						this_is_a_minimum = false
-- 					end
				else
					--print(eigen_value, "positive", math.abs(eigen_value / val_scale))
				end
			end
		end
	end
	

	return this_is_a_minimum
end

function writeStableSites(filename, mep)
	local f = io.open(filename, "w")
	
	timer1 = Timer.new()
	timer1:start()
	local dt=math.pi/32
	local n = 0
	for t1=0,math.pi+dt/2,dt do
		for t2=0,math.pi+dt/2,dt do
			n = n + 1
			if isStableConfiguration(mep, t1, t2) then
				f:write(table.concat({t1,t2,1}, "\t") .. "\n")
			end
		end
		f:write("\n")
	end
	f:close()
	timer1:stop()
	
	print("avrg query time: ", timer1:elapsed() / n)
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
	
	
	extra_plots = [[, "InitialComponents.dat" w lp pt 7 lt 1 lw 1.5 , "FinalComponents.dat" w lp pt 7 lt 3 lw 1.5,  "stable.dat" w p pt 6 lt 0 lw 1.5 ]]
	
		
	cmd = string.format([[
		#set term wxt size 1280,960
		set term pngcairo size 1280,960
		set output "map.png"
		set xrange [0:pi+1e-6]
		set yrange [0:pi+1e-6]
		set xlabel "Theta 1"
		set ylabel "Theta 2"
		
		set xrange [0:pi+1e-6]
		set yrange [0:pi+1e-6]
		set isosample 250, 250
		set table 'test.dat'
		splot "./energy_landscape.txt"
		unset table
		
		
		set contour base
		#set cntrparam level incremental -130, 0.33, 100
		set cntrparam levels auto 20
		#set cntrparam levels discrete -126,-122,-121,-120,-119.5,-119,-118,-114,-110
		unset surface
		set table "cont.dat"
		splot "./energy_landscape.txt"
		unset table
		
		reset
		set xrange [0:pi+1e-6]
		set yrange [0:pi+1e-6]
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

np = 32
mep = MEP.new()
mep:setSpinSystem(ss)
mep:setSites(sites)
mep:setInitialPath(initial_path)
mep:setEnergyFunction(energy)
mep:setNumberOfPathPoints(np)


mep:initialize(4)
mep:randomize(0.1) --adding a little noise to first guess
mep:initialize() -- so we can get the initial paths

writePathComponents("InitialComponents.dat")
-- recording initial 0th, 1st, 2nd and 3rd derivative of the energy
for i=0,3 do
	writeEnergyPath("InitialEnergyPath" .. i .. ".dat", mep, i)
end

requested_steps = {50,25}
non_default_tol = {0.5, 0.1}

for i=1,2 do
	successful_steps = mep:compute(requested_steps[i], non_default_tol[i]) 
	print(i, "Computed " .. successful_steps .. "/" .. requested_steps[i] .. " steps")
end

mins, maxs, all = mep:maximalPoints()
local stepCount = 15
local customTolerance = -1 --turning off adaptive stepper for point relax

for key,value in pairs(maxs) do
	print("max point steps:", mep:relaxSaddlePoint(value, stepCount, customTolerance))
end

for key,value in pairs(mins) do
	print("min point steps:", mep:relaxSinglePoint(value, stepCount, customTolerance))
end


writePathComponents("FinalComponents.dat")
-- recording final 0th, 1st, 2nd and 3rd derivative of the energy
for i=0,3 do
	writeEnergyPath("FinalEnergyPath" .. i .. ".dat", mep, i)
end


-- if false then
cmd = [[
	#set term wxt size 1024,768
	set term pngcairo size 1024,768
	set output "p%d.png"
	plot "./FinalEnergyPath%d.dat" w lp pt 7 lt 3 lw 1.5,  "./InitialEnergyPath%d.dat" w lp pt 7 lt 1 lw 1.5
]]

for i=0,3 do
	f = io.open("plot.cmd", "w")
	f:write(string.format(cmd, i, i, i))
	f:close()

	os.execute("gnuplot -persist plot.cmd")
end

writeStableSites("stable.dat", mep)

energy_landscape()
-- end


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

mins, maxs, all = mep:maximalPoints()
print("Curvature Data for maximal points")
for k,pidx in pairs(all) do
	point_data(pidx, mep)
end


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


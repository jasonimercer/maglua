dofile("maglua://POVRay.lua")

ss = SpinSystem.new(4,4,2)
zee = AppliedField.new(ss)
ani = Anisotropy.new(ss)
 ex = Exchange.new(ss)

-- setPeriodicXYZ is a new method, it's not required.
-- default periodic exchange values are true,true,true
ex:setPeriodicXYZ(true, true, false)
 
K, Jxy, Jz = 2, 0.3, 0.8
for k=1,ss:nz() do
	for j=1,ss:ny() do
		for i=1,ss:nx() do
			if k == ss:nz() then
				ss:setSpin({i,j,k}, {0,0,1}, 1/2)
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

zee:set({0.5, 0.1,-4.0}) -- canted field to break symmetry
zee:set({0, 0,-4.0}) -- canted field to break symmetry




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


function writeEnergyPath(filename)
	f = io.open(filename, "w")
	local ee = mep:pathEnergy()
	for i,value in ipairs(ee) do
		f:write(i .. "\t" .. value .. "\n")
	end
	f:close()
end


-- these are the sites involved in the
-- EB calculation. Order has meaning here.
sites = {{2,2,1}, {2,2,2}}

upup     = {{0,0, 1}, {0,0, 1/2}}
updown   = {{0,0, 1}, {0,0,-1/2}}
downdown = {{0,0,-1}, {0,0,-1/2}}
-- initial_path = {upup, updown, downdown}
initial_path = {upup, downdown}

np = 64

mep = MEP.new()
mep:setSpinSystem(ss)
mep:setSites(sites)
mep:setInitialPath(initial_path)
mep:setEnergyFunction(energy)
mep:setNumberOfPathPoints(np)
mep:setTolerance(1e-1)

mep:setTolerance(-1) --testing fixed step size
mep:setBeta(0.01)

-- initialize is not needed here, we're just doing it so we can get
-- the initial energy path to compare.
mep:initialize()
writeEnergyPath("InitialEnergyPath.dat")

mep:compute(200) -- this is the EB calculation


writeEnergyPath("FinalEnergyPath.dat")	

mins, maxs, all = mep:maximalPoints()
energies = mep:pathEnergy()

print("Energy Maximals after EB")
print("Mins:\nindex\tpath_idx\tenergy")
for k,pidx in pairs(mins) do
	print(k,pidx,energies[pidx])
end

print("Maxs:\nindex\tpath_idx\tenergy")
for k,pidx in pairs(maxs) do
	print(k,pidx,energies[pidx])
end

print("\nRelaxing minimum points into center of local basin")
mep:setTolerance(1e-2)

for k,pidx in pairs(mins) do
	mep:relaxSinglePoint(pidx, 10)
end
for k,pidx in pairs(maxs) do
	mep:relaxSaddlePoint(pidx, 10)
end

energies = mep:pathEnergy()
print("New Mins:")
print("index\tpath_idx\tenergy")
for k,pidx in pairs(mins) do
	print(k,pidx,energies[pidx])
end
print("New Maxs:")
print("index\tpath_idx\tenergy")
for k,pidx in pairs(maxs) do
	print(k,pidx,energies[pidx])
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

function curvature(idx)
	D = mep:hessianAtPoint(idx)
	print("Hessian at point " .. idx)
	printMat("D", D)

	vals, vecs = D:matEigen()

	print("Eigen Values at point " .. idx)
	printMat("evals", vals)

	print("Eigen Vectors at point " .. idx .. " (rows)")
	printMat("evecs", vecs)
	
	print()
end

-- print("Curvature Data for maximal points")
-- for k,pidx in pairs(all) do
-- 	curvature(pidx)
-- end



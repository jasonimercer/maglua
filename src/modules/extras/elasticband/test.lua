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

zee:set({0,0,-4.0})




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
	local ee = eb:pathEnergy()
	for i=1,table.maxn(ee) do
		f:write(i .. "\t" .. ee[i] .. "\n")
	end
	f:close()
end

function writePathComponents(filename)
	f = io.open(filename, "w")
	
	local np = eb:numberOfPathPoints()
	local ns = eb:numberOfSites()

	for p=1,np do
		local line = {}
		table.insert(line, p)
		for s=1,ns do
			local x, y, z = eb:spin(p, s)
			table.insert(line, x)
			table.insert(line, y)
			table.insert(line, z)
		end
		line = table.concat(line, "\t")
-- 		print(line)
		f:write(line .. "\n")
	end
	
	f:close()
end



-- these are the sites involved in the
-- EB calculation. Order has meaning here.
sites = {{2,2,1}, {2,2,2}}

upup     = {{0,0, 1}, {0,0, 1/2}}
updown   = {{0,0, 1}, {0,0,-1/2}}
downdown = {{0,0,-1}, {0,0,-1/2}}
initial_path = {upup, updown, downdown}

np = 64

eb = ElasticBand.new()
eb:setSites(sites)
eb:setInitialPath(initial_path)
eb:setEnergyFunction(energy)
eb:setNumberOfPathPoints(np)
eb:setSpinSystem(ss)


-- initialize is not needed here, we're just doing it so we can get
-- the initial energy path to compare.
eb:initialize()

writePathComponents("InitialComponents.dat")
writeEnergyPath("InitialEnergyPath.dat")

eb:compute(50)
writePathComponents("FinalComponents.dat")
writeEnergyPath("FinalEnergyPath.dat")	
	
-- the following will render the states
if false then
ss3 = ss:copy()
	for i=1,np do
		local fn = string.format("ss%04d.pov", i)
		eb:writePathPointTo(i,ss3)
		POVRay(fn, ss3,  {scale=0.5})
		os.execute("povray -D -W640 -H480 " .. fn .. "")
	end
end


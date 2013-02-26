dofile("maglua://POVRay.lua")

ss1 = SpinSystem.new(4,4,3)
ani = Anisotropy.new(ss1)
 ex = Exchange.new(ss1)

-- setPeriodicXYZ is a new method
ex:setPeriodicXYZ(true, true, false)
 
K, J, Jz = 2, 0.3, 0.8
for k=1,ss1:nz() do
	for j=1,ss1:ny() do
		for i=1,ss1:nx() do
			if k == ss1:nz() then
				ss1:setSpin({i,j,k}, {0,0,1}, 1/2)
			else
				ss1:setSpin({i,j,k}, {0,0,1}, 1)
			end
			
			ani:add({i,j,k}, {0,0,1}, K)
		
			ex:add({i,j,k}, {i+1,j,k}, J)
			ex:add({i,j,k}, {i-1,j,k}, J)
			ex:add({i,j,k}, {i,j+1,k}, J)
			ex:add({i,j,k}, {i,j-1,k}, J)
			ex:add({i,j,k}, {i,j,k-1}, Jz)
			ex:add({i,j,k}, {i,j,k+1}, Jz)
		end
	end
end

-- ss2 is the destination system, it is different from ss1 in some way
ss2 = ss1:copy()
ss2:setSpin({2,2,1}, {0,0,-1})
ss2:setSpin({2,2,2}, {0,0,-1})
ss2:setSpin({2,2,3}, {0,0,-1/2})

function energy(ss)
	ss:resetFields()
	ex:apply(ss)
	ani:apply(ss)
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
	local ns = table.maxn(eb:freeSites())

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
		print(line)
		f:write(line .. "\n")
	end
	
	f:close()
end

eb = ElasticBand.new()
eb:setInitialState(ss1)
eb:setFinalState(ss2)

local np = 64

-- these are the sites we will allow to change when moving from from ss1 to ss2
eb:setFreeSites({{2,2,1}, {2,2,2}, {2,2,3}})

eb:setEnergyFunction(energy)
eb:setNumberOfPathPoints(np)

-- initialize is not needed here, we're just doing it so we can get
-- the initial energy path to compare.
eb:initialize()
writePathComponents("InitialPathway.txt")

for i=1,3 do
	eb:compute(25)
	writePathComponents("FinalPathway" .. i .. ".txt")
end

-- the following will render the states
if false then
ss3 = ss1:copy()
	for i=1,np do
		local fn = string.format("ss%04d.pov", i)
		eb:writePathPointTo(i,ss3)
		POVRay(fn, ss3,  {scale=0.5})
		os.execute("povray -D -W640 -H480 " .. fn .. "")
	end
end

print("Done")

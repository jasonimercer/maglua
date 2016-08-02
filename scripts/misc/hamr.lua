-- Test hamr temperature/field pulse
--
-- command line arguments: argv
--    [1] maglua
--    [2] script
--    [3] temperature pulse: "std" or "hamr"
--    [4] random seed
--    [5] write start
--    [6] write step
--    [7] write end
--
-- example execution:
-- ./maglua  hamr.lua hamr 1000 2 0.1 10
-- 

nx, ny, nz = 10, 10, 10

ss      = SpinSystem.new(nx, ny, nz)
ex      = Exchange.new(nx, ny, nz)
field   = AppliedField.new(nx, ny, nz)
ani     = Anisotropy.new(nx, ny, nz)
thermal = Thermal.new(nx, ny, nz)
rng     = MTRand.new()
llg     = LLGCartesian.new()
seed    = argv[4]
rng:setSeed(tonumber(seed))

-- exchange neighbour directions
dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}

for k=1,nz do
	for j=1,ny do
		for i=1,nx do
			-- spins all up
			ss:setSpin(i, j, k, 0, 0, 1)

			--ZZ anisotropy
			ani:setSite(i, j, k, 0, 0, 1, 0.5)
			
			-- non-pbc J
			for n=1,6 do
				dx, dy, dz = i+dir[n][1], j+dir[n][2], k+dir[n][3]
				if ex:member(dx, dy, dz) then
					ex:addPath(i, j, k, dx, dy, dz, 1.0)
				end
			end
		end
	end
end



function gaussian(height, mean, fwhm, x)
	stddev = fwhm / (2 * math.sqrt(2 * math.log(2)))
	return height * math.exp( - (x-mean)*(x-mean) / (2*stddev*stddev))
end

function savestate(filename, spinsystem)
	output = io.open(filename, "w")
	
	for k=1,spinsystem:nz() do
		for j=1,spinsystem:ny() do
			for i=1,spinsystem:nz() do
				local x, y, z = spinsystem:spin(i,j,k)
				data = string.format("%i %i %i %f %f %f\n", i ,j, k, x, y, z)
				--output:write(x, "\t,", y, "\t,", z, "\n")
				output:write(data)
			end
		end
	end
	
	output:close()
end



dt      =  0.01
bmax    = -0.6
alpha   =  1.0
maxtime =  20

if argv[3] == "hamr" then
	Tmax = 2.0
	Tmin = 0.5
	fn = "hamr-" .. seed .. ".dat"
else
	if argv[3] == "std" then
		Tmax = 0.5
		Tmin = 0.5
		fn = "std-" .. seed .. ".dat"
	else
		error("require argument 'hamr' or 'std'")
	end
end




print(fn)
f = io.open(fn, "w")

llg:setAlpha(alpha)
llg:setTimeStep(dt)

writestart = tonumber(argv[5]) --convert command line string to number
writestep  = tonumber(argv[6])
writeend   = tonumber(argv[7])
writeprevious = 0

print(writestart, writestep, writeend)

while ss:time() < maxtime do
	ss:zeroFields()

	t = gaussian(Tmax - Tmin, 5, 3, ss:time()) + Tmin
	b = gaussian(bmax, 8, 6, ss:time())

	thermal:setTemperature(t)
	field:set(0,0,b)

	ex:apply(ss)
	field:apply(ss)
	ani:apply(ss)
	thermal:apply(llg, rng, ss)

	ss:sumFields()

	llg:apply(ss)

	x, y, z, m = ss:netMag()

	f:write(ss:time(), "\t", t, "\t", b, "\t", z/1000, "\t", m/1000, "\n")

-- 	if ss:time() >= writestart - dt/2 and ss:time() < writeend then
-- 		if ss:time() >= writeprevious + writestep-dt/2 then
			filename = string.format("state-%06.3f.dat", ss:time())
			print(filename)
-- 			writeprevious = ss:time()
			savestate(filename, ss)
-- 		end
-- 	end

end

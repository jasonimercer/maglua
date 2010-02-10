nx, ny, nz = 10, 10, 10

ss      = SpinSystem.new(nx, ny, nz)
ex      = Exchange.new(nx, ny, nz)
field   = AppliedField.new(nx, ny, nz)
ani     = Anisotropy.new(nx, ny, nz)
thermal = Thermal.new(nx, ny, nz)
rng     = MTRand.new()

llgc = LLGCartesian.new()
llgq = LLGQuaternion.new()
llg  = llgc

rng:setSeed(11112)

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
g = gaussian


alpha = 1.0
dt = 0.01
maxtime = 20

bmax = -0.6

if argv[3] == "hamr" then
	Tmax = 2.0
	Tmin = 0.5
	fn = "hamr.dat"
else
	if argv[3] == "std" then
		Tmax = 0.5
		Tmin = 0.5
		fn = "std.dat"
	else
		error("require argument 'hamr' or 'std'")
	end
end

print(fn)
f = io.open(fn, "w")

llgc:setAlpha(alpha)
llgc:setTimeStep(dt)

llgq:setAlpha(alpha)
llgq:setTimeStep(dt)

while llg:time() < maxtime do
	ss:zeroFields()

	t = g(Tmax - Tmin, 5, 2, llg:time()) + Tmin
	b = g(bmax, 8, 6, llg:time())

	thermal:setTemperature(t)
	field:set(0,0,b)


	ex:apply(ss)
	field:apply(ss)
	ani:apply(ss)
	thermal:apply(llg, rng, ss)

	ss:sumFields()

	llg:apply(ss)

	x, y, z, m = ss:netMag()

	f:write(llg:time(), "\t", t, "\t", b, "\t", z/1000, "\n")
end

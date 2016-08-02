nx, ny, nz = 10, 10, 10

ss      = SpinSystem.new(nx, ny, nz)
ex      = Exchange.new(nx, ny, nz)
field   = AppliedField.new(nx, ny, nz)
ani     = Anisotropy.new(nx, ny, nz)
thermal = Thermal.new(nx, ny, nz)
rng     = MTRand.new()

llgc = LLGCartesian.new()
llgq = LLGQuaternion.new()


field:set(0,0,1.0)
rng:setSeed(11112)
thermal:setTemperature(1.0)


-- exchange neighbour directions
dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}

for k=1,nz do
	for j=1,ny do
		for i=1,nx do
			--spins in random directions
			x, y, z = rng:normal(), rng:normal(), rng:normal()
-- 			x, y, z = 1, 0, 0
			isl = 1.0/math.sqrt(x*x + y*y + z*z)
			x, y, z = x*isl, y*isl, z*isl
			ss:setSpin(i, j, k, x, y, z)

			--ZZ anisotropy
			ani:setSite(i, j, k, 0, 0, 1, 1)
			
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

llg, fn = nil, nil
if argv[3] == "c" then
	llg = llgc
	fn  = "datac.dat"
else
	if argv[3] == "q" then
		llg = llgq
		fn  = "dataq.dat"
	else
		error("argv[3] is not c or q")
	end
end


alpha = 0.5
dt = 0.01

print(fn)
f = io.open(fn, "w")

llgc:setAlpha(alpha)
llgc:setTimeStep(dt)

llgq:setAlpha(alpha)
llgq:setTimeStep(dt)

while llg:time() < 10 do
	ss:zeroFields()

	ex:apply(ss)
	field:apply(ss)
	ani:apply(ss)
	thermal:apply(llg, rng, ss)

	ss:sumFields()

	llg:apply(ss)

	x, y, z, m = ss:netMag()

	f:write(llg:time(), "\t", m, "\n")
end

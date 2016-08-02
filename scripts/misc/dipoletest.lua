nx, ny, nz = 50, 50, 1

ss      = SpinSystem.new(nx, ny, nz)
dip     = Dipole.new(nx, ny, nz)
field   = AppliedField.new(nx, ny, nz)
ex      = Exchange.new(nx, ny, nz)
thermal = Thermal.new(nx, ny, nz)

llg     = LLGQuaternion.new()
rng     = MTRand.new()

field:set(1,0,0)

rng:setSeed(1234)
thermal:setTemperature(0.0)

dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}

for k=1,nz do
	for j=1,ny do
		for i=1,nx do
			--spins in random directions
			x, y, z = rng:normal(), rng:normal(), rng:normal()
			isl = 1.0/math.sqrt(x*x + y*y + z*z)
			x, y, z = x*isl, y*isl, z*isl
			ss:setSpin(i, j, 1, x, y, z)

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

fn = "dipole.dat"
alpha = 1.0
dt = 0.05

print(fn)
f = io.open(fn, "w")

llg:setAlpha(alpha)
llg:setTimeStep(dt)

while ss:time() < 5 do
	ss:zeroFields()

	ex:apply(ss)
	dip:apply(ss)
	field:apply(ss)
	-- 	thermal:apply(llg, rng, ss)

	ss:sumFields()

	llg:apply(ss)

-- 	x, y, z = 0, 0, 0
-- 	for i=1,10,2 do
-- 		for j=1,10,2 do
-- 			sx, sy, sz = ss:spin(i,j,1)
-- 			x, y, z = x+sx, y+sy, z+sz
-- 		end
-- 	end

	x, y, z, m = ss:netMag()
-- 	m = math.sqrt(x^2+y^2+z^2) / 25

	f:write(ss:time(), "\t", m, "\n")
end
f:close()

g = io.open("config.dat", "w")
for k=1,nz do
	for j=1,ny do
		for i=1,nx do
			g:write(string.format("%i %i %i %g %g %g\n", i, j, k, ss:spin(i,j,k)))
		end
	end
end
g:close()


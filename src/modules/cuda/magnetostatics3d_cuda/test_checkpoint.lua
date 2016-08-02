-- uMAG Standard Problem #4
-- Generate Initial S-State

dofile("maglua://CGS.lua")

-- cell size
a, b, c = 5*nm, 5*nm, 3*nm
cell = a*b*c

-- system size
dx, dy, dz = 500*nm, 125*nm, 3*nm

nx, ny, nz = dx/a, dy/b, dz/c
print(nx,ny,nz)

-- making SpinSystem 2x required size so we
-- can use FFTs + truncated interactions
ss = SpinSystem.new(nx*2, ny*2, nz*2)

mag = Magnetostatics3D.new(ss)
-- mag:setTruncation(1000, ss:nx()/2-1, ss:ny()/2-1, ss:nz()/2-1) 
mag:setTruncation(1000, 1,1,1) 
mag:setUnitCell({a,0,0}, {0,b,0}, {0,0,c})
mag:setGrainSize(a,b,c)
mag:setStrength((4*math.pi)/(cell))

mag:apply(ss)


checkpointSave("foo.dat", mag, mag, ss)

			
			
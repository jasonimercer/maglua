-- This file will create 3D dipole interaction tensors

nx, ny, nz = 24, 24, 24

lr = LongRange3D.new(nx,ny,nz)
ewald = DipoleEwald3D.new(nx,ny,nz)

-- cubic
--a = {1,0,0}
--b = {0,1,0}
--c = {0,0,1}
--desc = "cubic"

-- fcc
a = {1,0,0}
b = {1/2, (3/4)^(1/2), 0}
c = {1/2, (3^(1/2))/6, (2/3)^(1/2)}
desc = "fcc"

ewald:setUnitCell(a, b, c)


ab = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"}

mat = {}
for i=1,6 do
	print("Generating " .. ab[i] .. " elements")
	for z=0,nz-1 do
		for y=0,ny-1 do
			for x=0,nx-1 do
				-- Note the -1 factor here
 				local value = -1 * ewald:calculateTensorElement(ab[i], {x,y,z}) 
				lr:setMatrix(ab[i], {x,y,z}, value)
			end
		end
	end
end

filename = string.format("%dx%dx%d_%s.lua", nx, ny, nz, desc)
lr:saveTensors(filename)

print("Tensor saved to `" .. filename .. "' have a look at it, it's human readable")

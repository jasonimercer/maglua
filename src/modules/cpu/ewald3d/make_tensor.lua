nx, ny, nz = 16, 16, 16

lr = LongRange3D.new(nx,ny,nz)
ewald = DipoleEwald3D.new(nx,ny,nz)
ewald:setUnitCell({1,0,0}, {0,1,0}, {0,0,1})

ab = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"}

for i=1,6 do
	for z=0,nz-1 do
		for y=0,ny-1 do
			for x=0,nx-1 do
				-- -1 due to conventions in MagLua
				local t = -1.0 * ewald:calculateTensorElement(ab[i], {x,y,z}) 
				print(ab[i], x,y,z, t)
				
				lr:setMatrix(ab[i], {x,y,z}, t)
			end
		end
	end
end

lr:saveTensors("16x16x16_cubic.lua")


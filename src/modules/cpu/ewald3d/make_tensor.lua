nx, ny, nz = 8, 8, 8

lr = LongRange3D.new(nx,ny,nz)
ewald = DipoleEwald3D.new(nx,ny,nz)
ewald:setUnitCell({1,0,0}, {0,1,0}, {0,0,1})

ab = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"}

mat = {}
for i=1,6 do
	for z in mpi.range(0,nz-1) do
		for y=0,ny-1 do
			for x=0,nx-1 do
 				local value = -1 * ewald:calculateTensorElement(ab[i], {x,y,z}) 
				table.insert(mat, {ab[i], x,y,z, value})
			end
		end
	end
end

mat = mpi.gather(1, mat)

if mpi.get_rank() == 1 then
	for i=1,mpi.get_size() do
		for k,v in pairs(mat[i]) do
			lr:setMatrix(v[1], {v[2],v[3],v[4]}, v[5])
		end
	end
	
	lr:saveTensors(string.format("%dx%dx%d_cubic.lua", nx, ny, nz))
end

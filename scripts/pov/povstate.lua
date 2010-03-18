dofile("colormap.lua")
dofile("povspin.lua")


loc = {18,14,18}
 at = {5.5,4.2,5.5}
-- loc = {12*10,12*10,-12*10}
loc = {25,10,-25}
 at = {0,0,0}

filename = io.read("*line")
n = "*number"

function range(t,i)
	local _min = t[1][i]
	local _max = t[1][i]

	for k,v in pairs(t) do
		if v[i] < _min then
			_min = v[i]
		end
		if v[i] > _max then
			_max = v[i]
		end
	end
	return _min, _max
end


while filename do
	print(filename)

	file = io.open(filename, "r")

	spins = {}
	c = 0
	
	i, j, k, x, y, z = file:read(n,n,n,n,n,n)
	while i do
		local tn = tonumber
		c = c + 1
		spins[c] = {i, k, j, x, y, z}
		i, j, k, x, y, z = file:read(n,n,n,n,n,n)
	end
	file:close()

	xmin, xmax = range(spins, 1)
	ymin, ymax = range(spins, 2)
	zmin, zmax = range(spins, 3)

	at = {xmin + 0.5*(xmax-xmin),
		  ymin + 0.4*(ymax-ymin),
		  zmin + 0.5*(zmax-zmin)}
	
	print(ymax, ymin)

	local xx, yy, zz = 0.5*(xmax+xmin), 0.5*(ymax+ymin), 0.5*(zmax+zmin)

	yy = xx
	loc = {-1*xx, 2.5*yy, -1*zz}
	lightpos = {{-8*xx, 4*yy, zz}, {-8*xx, 8*yy, -8*zz}, {xx, 4*yy, -8*zz}}

	pov  = io.open(filename .. ".pov", "w")
	pov:write( povprefix(loc, at, lightpos) )

	for k,v in pairs(spins) do
		r, g, b = colormap(v[4], v[5], v[6]) 
		pov:write(spin(v[4], v[5], v[6], v[1], v[2], v[3], r, g, b))
	end
	pov:close()

	
	filename = io.read("*line")
end


		


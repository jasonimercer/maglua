dofile("maglua://ColorMap.lua")

if true then
	XX = Dipole.XX
	XY = Dipole.XY
	XZ = Dipole.XZ

	YY = Dipole.YY
	YZ = Dipole.YZ

	ZZ = Dipole.ZZ
else
	XX = Magnetostatics2D.PXX
	XY = Magnetostatics2D.PXY
	XZ = Magnetostatics2D.PXZ

	YY = Magnetostatics2D.PYY
	YZ = Magnetostatics2D.PYZ

	XX = Magnetostatics2D.PZZ
end

local function field(b, m, rx,ry,rz)
	local xx = XX(rx,ry,rz,b)
	local xy = XY(rx,ry,rz,b)
	local xz = XZ(rx,ry,rz,b)

	local yy = YY(rx,ry,rz,b)
	local yz = YZ(rx,ry,rz,b)

	local zz = ZZ(rx,ry,rz,b)

	return 	m[1]*xx + m[2]*xy + m[3] * xz,
			m[1]*xy + m[2]*yy + m[3] * yz,
			m[1]*xz + m[2]*yz + m[3] * zz
end




width, height = 320, 320
png = PNGWriter.new("test.png", width, height)
b = {2,1,1}

-- font = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
-- png:plot_text(font, 16, 10, 30, 0, "Hello",  0,0,1)

data = {}
for x=1,width do
	data[x] = {}
	for y=1,height do
		data[x][y] = {0,0,0}
	end
end


dd = {{1,1}, {-1,1}, {-1,-1}, {1,-1}}

local pi = math.pi
m = {{1,0,0}, {-1,0,0}, {1,0,0}, {math.cos(pi/8), math.sin(pi/8), 0}}

for x=1,width do
	for y=1,height do
		
		for j=1,4 do
			local rx = (x-width/2) * 0.03 + dd[j][1]*2 + b[1]/2
			local ry = (y-height/2) * 0.03 + dd[j][2]*2 + b[2]/2
			local rz = b[3]/2
			local hx,hy,hz = field(b, m[j], rx,ry,rz)
			
			data[x][y][1] = data[x][y][1] + hx
			data[x][y][2] = data[x][y][2] + hy
			data[x][y][3] = data[x][y][3] + hz
		end

		local hx, hy, hz = data[x][y][1], data[x][y][2], data[x][y][3]
		local r,g,b = ColorMap(hx,hy,hz)
		
		local hh = math.sqrt(hx*hx + hy*hy + hz*hz)
		
		local m = math.cos(hh*10) * 0.4 + 0.6
		
		
		png:plot(x, y, r*m, g*m, b*m)

	end
end


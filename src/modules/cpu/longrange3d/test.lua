n = 6
ss1 = SpinSystem.new(n,n,n)

ex = Exchange.new(ss1)
lr3d = LongRange3D.new(ss1)

llg = LLG.Cartesian.new()

ss1:setTimeStep(5e-3)
J = 2
for z=1,n do
	for y=1,n do
		for x=1,n do
			ss1:setSpin({x,y,z}, {1,0,0})
			
			ex:add({x,y,z}, {x+1,y,z}, J)
			ex:add({x,y,z}, {x-1,y,z}, J)
			ex:add({x,y,z}, {x,y+1,z}, J)
			ex:add({x,y,z}, {x,y-1,z}, J)
			ex:add({x,y,z}, {x,y,z+1}, J)
			ex:add({x,y,z}, {x,y,z-1}, J)
		end
	end
end

ss1:setSpin({1,1,1}, {0,0,1})
ss2 = ss1:copy()

-- building 3D long range to act like PBC3D exchange
cart = {{1,0,0}, {-1,0,0}, 
		{0,1,0}, {0,-1,0}, 
		{0,0,1}, {0,0,-1}}

for i=1,6 do
	lr3d:setMatrix("XX", cart[i], J)
	lr3d:setMatrix("YY", cart[i], J)
	lr3d:setMatrix("ZZ", cart[i], J)
end

function step1()
	ss1:resetFields()
	
	ex:apply(ss1)
	ss1:sumFields()
	
	llg:apply(ss1)
end

function step2()
	ss2:resetFields()
	
	lr3d:apply(ss2)
	ss2:sumFields()
	
	llg:apply(ss2)
end

f = io.open("test.dat", "w")
sample = {4,3,4}
while ss1:time() < 10.0 do
	for i=1,50 do
		step1()
		step2()
	end
	
	local x1, y1, z1 = ss1:spin(sample)
	local x2, y2, z2 = ss2:spin(sample)
	
	f:write(table.concat({x1,x2,y1,y2,z1,z2}, "\t") .. "\n")
end
f:close()

f = io.open("plot.txt", "w")
f:write([[
	set title "If the data below match then things are working"
	plot "test.dat" using 1 title "Exchange Method - x component" w l, "test.dat" using 2 title "LongRange3D Method - x component" w p
]])
f:close()
os.execute("gnuplot -persist plot.txt")


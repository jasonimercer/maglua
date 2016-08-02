bsp = BSPTree.new(0,0,0,    10,10,10)

for i=1,10 do
	x = math.random() * 8 + 1
	y = math.random() * 8 + 1
	z = math.random() * 8 + 1

	bsp:insert(x, y, z, {i, i+i, i+i+i})
end

bsp:split()

t = bsp:getDataInSphere(5,5,5, 2)

for k,v in pairs(t) do
	print(k,v)
end




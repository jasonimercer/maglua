ss = SpinSystem.new(4,4)
mp = Multipole.new(ss)

for i=1,4 do
	for j=1,4 do
		print(mp:getPosition({i,j}))
	end
end


mp:preCompute()
--L = 6
--x = 1/2

--print("L = " .. L .. ", x = " .. x)

--for m=-5,4 do
--	print(L,m,Plm(L,m,x))
--end

m = Multipole.new()

t = m:i2i(0.5, 0.1, 3/4)

n = 9
for r=1,n do
	for c=1,n do
		t[r][c] = string.format("% 4.2f % 4.2fi", t[r][c][1], t[r][c][2])
	end
	print(table.concat(t[r], "\t"))
end

-- n = 10
-- 
-- px = Array.Double.new(n)
-- py = Array.Double.new(n)
-- pz = Array.Double.new(n)
-- 
-- sx = Array.Double.new(n)
-- sy = Array.Double.new(n)
-- sz = Array.Double.new(n)
-- 
-- for i=1,n do
-- 	local x, y, z = math.random(), math.random(), math.random()
-- 	print(string.format("%5f  %5f  %5f", x,y,z))
-- 	px:set(i, x)
-- 	py:set(i, y)
-- 	pz:set(i, z)
-- 
-- 	sx:set(i, math.random()*2-1)
-- 	sy:set(i, math.random()*2-1)
-- 	sz:set(i, math.random()*2-1)
-- end
-- 
-- octtree = OctTree.new(px,py,pz, sx,sy,sz)
-- octtree:split()


-- print("Parent: ", octtree:count())
-- for i=1,8 do
-- 	print("Child" .. i, octtree:child(i):count())
-- end
-- 
-- print("Parent Bounds:")
-- low, high = octtree:bounds()
-- 
-- print(string.format("%5f  %5f  %5f",  low[1],  low[2],  low[3]))
-- print(string.format("%5f  %5f  %5f", high[1], high[2], high[3]))
-- 
-- print("Child Bounds:")
-- for i=1,8 do
-- 	low, high = octtree:child(i):bounds()
-- 
-- 	print(i, string.format("%5f  %5f  %5f  -  %5f  %5f  %5f",  low[1],  low[2],  low[3], high[1], high[2], high[3]))
-- 	for j=1,3 do
-- 		print(high[j] - low[j])
-- 	end
-- 	
-- -- 	print(octtree:child(i):child(1))
-- 	
-- end




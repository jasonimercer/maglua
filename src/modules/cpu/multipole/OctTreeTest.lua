max_degree = 2
n = 2

px = Array.Double.new(n)
py = Array.Double.new(n)
pz = Array.Double.new(n)

sx = Array.Double.new(n)
sy = Array.Double.new(n)
sz = Array.Double.new(n)


px:set(1, 1)
py:set(1, 2)
pz:set(1, 3)

if n > 1 then
	px:set(2, 1)
	py:set(2, 2)
	pz:set(2, 3.5)
end

for i=1,n do
	sx:set(i,(1/2)^(1/2))
	sy:set(i, 0)
	sz:set(i,(1/2)^(1/2))
end


function printNode(name, node)
	local c = node -- octtree:child(i)
	print(name .. ": " .. c:count() .. " points")


	if c:count() > 0 then
		local t = node:innerTensor()
		for r=1,table.maxn(t) do
			t[r] = string.format("%6.4f %6.4fi", t[r][1], t[r][2])
		end
		table.insert(t, 5, "|")
		table.insert(t, 2, "|")
		print(table.concat(t, "  "))
	
		local o = c:localOrigin()
		print("local origin: ", table.concat(o, ", "))

		for j=1,c:count() do
			local k = c:member(j)
			local x = px:get(k)
			local y = py:get(k)
			local z = pz:get(k)
			print("point " .. k,"global",x,y,z)
			print("point " .. k,"local",x-o[1], y-o[2], z-o[3])
		end
	end
	print()
end


octtree = FMMOctTree.new(max_degree, px,py,pz, sx,sy,sz)

do_split = true

if do_split then
	octtree:split(1)
	octtree:calcInnerTensor();
	
	printNode("root", octtree)
	for i=1,8 do
			printNode("child" .. i, octtree:child(i))
	end
else
	octtree:calcInnerTensor();
	printNode("root", octtree)
end
	
	
	


--[[
-- a, b = octtree:bounds()
o = octtree:localOrigin()

-- print(px:get(1), py:get(1), pz:get(1))
-- 
-- print(table.concat(a, ", "))
print(table.concat(o, ", "))
-- print(table.concat(b, ", "))

octtree:calcInnerTensor(1e-6)
-- 
t = octtree:innerTensor()
for r=1,table.maxn(t) do
	t[r] = string.format("%f %fi", t[r][1], t[r][2])
end
print("Root Inner Tensor (" .. octtree:count() .. "):\n" .. table.concat(t, "\n"))
--]]

-- hx, hy, hz = octtree:fieldAt({8,0,0})
-- 
-- -- print(hx,hy,hz)
-- 
-- 
-- if octtree:child(1) then
-- 
-- 	hx, hy, hz = 0,0,0
-- 	for i=1,8 do
-- 		x,y,z = octtree:child(i):fieldAt({8,0,0})
-- 		hx, hy, hz = hx+x, hy+y, hz+z
-- 	end
-- 
-- -- 	print(hx,hy,hz)
-- end


-- t = octtree:innerTensor()
-- for r=1,table.maxn(t) do
-- 	t[r] = string.format("% 6.3f % 6.3fi", t[r][1], t[r][2])
-- end
-- print("Inner Tensor Parent (" .. octtree:count() .. "):  " .. table.concat(t, "\t"))
-- 	
-- for i=1,8 do
-- 	c = octtree:child(i)
-- 	
-- 	t = c:innerTensor()
-- 	
-- 	for r=1,table.maxn(t) do
-- 		t[r] = string.format("% 6.3f % 6.3fi", t[r][1], t[r][2])
-- 	end
-- 	print("Inner Tensor Child[" .. i .. "] (" .. c:count() .. "): " .. table.concat(t, "\t"))
-- end

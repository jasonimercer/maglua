function printMat(name, M)
	local default = {"  " .. name, "(" .. M:ny() .. "x" .. M:nx() .. ")"}
	for r=1,M:ny() do
		local t = {default[r] or ""}
		for c=1,M:nx() do
			table.insert(t, string.format("% 06.6f", M:get(c,r)))
		end
		print(table.concat(t, "\t"))
	end
	print()
end

A = Array.Double.new(4,4)
x = Array.Double.new(1,4)

A:setFromTable({{8,2,-3,4}, {2,14,6,-1}, {-3,6,12,-5}, {4,-1,-5, 7}})
x:setFromTable({{9},{3},{5},{1}})

A:setFromTable({{8,2,-3,4}, {2,14,6,-1}, {-3,6,12,-5}, {4,-1,-5, 7}})
x:setFromTable({{9},{3},{5},{1}})

printMat("A", A)
printMat("x", x)

b = A:matMul(x)
printMat("b", b)

y = A:matInv():matMul(b)

printMat("y", y)


-- Expected Output
--   A	 8.000000	 2.000000	-3.000000	 4.000000
-- (4x4) 2.000000	 14.000000	 6.000000	-1.000000
-- 		-3.000000	 6.000000	 12.000000	-5.000000
-- 		 4.000000	-1.000000	-5.000000	 7.000000
-- 
--   x	 9.000000
-- (4x1) 3.000000
-- 		 5.000000
-- 		 1.000000
-- 
--   b	 67.000000
-- (4x1) 89.000000
-- 		 46.000000
-- 		 15.000000
-- 
--   y	 9.000000
-- (4x1) 3.000000
-- 		 5.000000
-- 		 1.000000

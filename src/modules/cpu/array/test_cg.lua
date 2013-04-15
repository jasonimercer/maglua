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

printMat("A", A)
printMat("x", x)

b = A:matMul(x)
printMat("b", b)


function conjGrad(A, b, tol)
	local tol = tol or 1e-8
	local x = b:copy()
		
	x:setAll(1) --initial guess
	local Ax = A:matMul(x)
	
	local r = b:pairwiseScaleAdd(-1, Ax) -- residual: r = b - A*x
	local p = r:copy() -- p = r
	local rs_old = r:dot(r) -- rs_old = r^T r
	
	local Ap = p:copy()
	
	for i=1,1000 do
		A:matMul(p, Ap)
		
		local alpha = rs_old / p:dot(Ap)
		x:pairwiseScaleAdd( alpha,  p, x) -- x = x + alpha p
		r:pairwiseScaleAdd(-alpha, Ap, r) -- r = r - alpha (A p)
		
		local rs_new = r:dot(r)
		
		if rs_new < tol*tol then
			return x
		end
		
		r:pairwiseScaleAdd(rs_new/rs_old, p, p) -- p = r + rs_new/rs_old p
		rs_old = rs_new
	end
	
	error("Failed to converge")
end

y = conjGrad(A, b)

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
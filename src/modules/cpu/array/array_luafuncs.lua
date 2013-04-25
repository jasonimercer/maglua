local mm = {"Array.Double", "Array.Float"}
local tolt = {1e-10, 1e-5}

for k,v in pairs(mm) do
	local MODNAME = v
	local MODTAB = loadstring("return " .. v)()
	local _tol = tolt[k]
	--local MODTAB = Array.Double
	local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time


	local matrix_invert


	local function matrix_invert(A, B)
		if A:nx() ~= A:ny() then
			error("Cannot invert non-square matrix")
		end

		if B == nil then
			B = A:copy()
			B:zero()
		end

		if B:nx() ~= A:nx() or B:ny() ~= B:nx() then
			error("Destination matrix size not equal to input matrix size")
		end

		local detA = A:matDet()

		if detA == 0 then
			return nil
		end

		if A:nx() == 1 then
			B:set({1}, 1/detA)
			return B
		end

		if A:nx() == 2 then
			B:set({1,1}, A:get({2,2}))
			B:set({2,1},-A:get({2,1}))
			B:set({1,2},-A:get({1,2}))
			B:set({2,2}, A:get({1,1}))
			B:scale(1/detA)
			return B
		end
		
		-- matrix is bigger than 2 so we'll use the blockwise inversion
		-- [ a b ] -1
		-- [ c d ]
		--
		-- =
		--
		-- [ inv(a) + inv(a) b inv(d - c inv(a) b) c inv(a)        - inv(a) b inv(d - c inc(a) b) ]
		-- [      -inv(d - c inv(a) b) c inv(a)                         inv(d - c inv(a) b )      ]

		local nx, ny = A:nx(), A:ny()
		local q = math.ceil(nx/2)
		local r = q + 1
		
		local rA = {{1,1}, { q, q}}
		local rB = {{r,1}, {nx, q}}
		local rC = {{1,r}, { q,ny}}
		local rD = {{r,r}, {nx,ny}}

		local a = A:slice(rA)
		local b = A:slice(rB)
		local c = A:slice(rC)
		local d = A:slice(rD)

		local inva = matrix_invert(a)
		local c_inva = c:matMul(inva)
		local c_inva_b = c_inva:matMul(b)

		local d_c_inva_b = d:pairwiseScaleAdd(-1, c_inva_b)

		local inv_d_c_inva_b = matrix_invert(d_c_inva_b)

		local inva_b = inva:matMul(b)

		local inva_b_inv_d_c_inva_b = inva_b:matMul(inv_d_c_inva_b)
		local inva_b_inv_d_c_inva_b_c = inva_b_inv_d_c_inva_b:matMul(c)
		local inva_b_inv_d_c_inva_b_c_inva = inva_b_inv_d_c_inva_b_c:matMul(inva)
		local AA = inva:pairwiseScaleAdd(1, inva_b_inv_d_c_inva_b_c_inva)

		--  - inv(a) b inv(d - c inc(a) b)
		local BB = inva_b_inv_d_c_inva_b:copy()
		BB:scale(-1)

		
		-- -inv(d - c inv(a) b) c inv(a)
		local inv_d_c_inva_b_c = inv_d_c_inva_b:matMul(c)
		local inv_d_c_inva_b_c_inva = inv_d_c_inva_b_c:matMul(inva)
		local CC = inv_d_c_inva_b_c_inva:copy()
		CC:scale(-1)


		--  inv(d - c inv(a) b )

		local DD = inv_d_c_inva_b:copy()

		
		-- AA BB
		-- CC DD

		AA:slice({{1,1}, {q,q}}, B, {{1,1},{q,q}})
		BB:slice({{1,1}, {nx-q,q}}, B, rB)
		CC:slice({{1,1}, {q,ny-q}}, B, rC)
		DD:slice({{1,1}, {nx-q,ny-q}}, B, rD)

		return B
	end



	local function qr_decompose_step(A, Q)
		local x = A:slice({{1,1}, {1,A:ny()}})
		
		local alpha = math.sqrt(x:dot(x))
		
		-- alpha gets the negative sign of x_k
		if x:get({1,1}) > 0 then
			alpha = -alpha
		end
		
		local ek = x:copy()
		ek:zero()
		ek:set({1,1}, 1)
		
		local u = x:pairwiseScaleAdd(alpha, ek)
		
		local v = u:copy()
		
		v:scale( 1/(math.sqrt(u:dot(u))))
		
		if Q == nil then
			Q = A:copy()
		end

		
		local vvT = v:matMul( v:matTrans())
		
		Q:matMakeI()
		Q:pairwiseScaleAdd(-2, vvT, Q)
		
		return Q
	end


	local function qr_decompose(_A, Q, R)
		local A = _A:copy()
		Q = Q or A:copy()
		R = R or A:copy()
		local T = A:copy()
		
		Q:matMakeI()
		R:matMakeI()
		
		Qt = {}
		
		local n = A:nx()
		local m = n-1
		for k=1,m do
			local sub = A:slice({{k,k}, {n,n}})
			Qk = qr_decompose_step(sub)
			
			R:matMakeI()
			Qk:slice({{1,1},{Qk:nx(),Qk:ny()}}, R, {{k,k}, {n,n}})
			-- R now holds the filled out Qk

			Qt[k] = R:copy()

			R:matMul(A, T)
			T:copy(A)
		end

		Q:matMakeI()
		for k=1,m do
			local qq = Qt[k]:matTrans()
			Q:copy(T)
			
			T:matMul(qq, Q)
		end
		
		Q:matTrans(T)
		
		T:matMul(_A, R)
		
		return Q, R
	end

	-- need to figure this out
	local function matEigenSystemFromSchur(A,U)
		local xx = {{1,1}, {1,A:ny()}}
		local ws = A:slice(xx)
		local M = A:copy()
		-- need to diagonalize A and update U
		for c=A:nx(),2,-1 do
			for r=1,c-1 do
	-- 			print(r,c)
	-- 			printMat("D", A)
				local frac = A:get(c,r) / A:get(c,r+1)
				M:matMakeI()
				M:set(r+1,r, -frac)
				
				M:matMul(A):copy(A)
				
	-- 			M:matInv():copy(M)
				
	-- 			M:matMul(U):copy(U)
				
				M:matTrans():matMul(U:matTrans()):matTrans():copy(U)
				
	-- 			print("frac", frac)
	-- 			for i=r+1,A:nx() do
	-- 			for i=1,A:nx() do
	-- 				print(i,"val = ", A:get(i,r), " delta = ", frac * A:get(i,r+1))
	-- 				A:set(i,r, A:get(i,r) - frac * A:get(i,r+1))
	-- 				U:set(i,r, U:get(i,r) - frac * U:get(i,r+1))
	-- 				U:set(r,i, U:get(r,i) - 1/frac * U:get(r+1,i))
	-- 			end
			end
		end
			
		for c=1,A:nx() do
			local yy = {{c,1}, {c,A:ny()}} 
			U:slice(yy, ws, xx)
			ws:scale(1/( ws:dot(ws)^(1/2) ))
			ws:slice(xx, U, yy)
		end
		
		return A,U
	end

	-- return a diagonal matrix with eigen values
	-- and a full matrix with columns on eigen vectors
	local function matEigenValues(_A, tol)
		local A = _A:copy()
		local tol = tol or _tol
		
		local Q, R = nil, nil
		
		local T = A:copy()
		T:matMakeI()
		
		local lastSum = 1
		local count_down = 5
		for i=1,10000 do
			Q,R = qr_decompose(A, Q, R)
			
			local b = T:matMul(Q)
			b:copy(T)
			
			R:matMul(Q, A)
			
			local thisSum = R:sum()
			
			local foo = (lastSum^2 - thisSum^2)^(2)

			if foo < tol then
	-- 			print(i)
				return A,T
			else
	-- 			print(foo, tol)
			end

	-- 		printMat("A", A)
			A:matLower(R)
	-- 		printMat("R", R)
	-- 		print(R:dot(R), tol)
	-- 		print( R:dot(R), tol)
			if R:dot(R) < tol then -- schur
	-- 			printMat("R", R)
	-- 			printMat("A", A)
	-- 			printMat("T", T)
	-- 			return matEigenSystemFromSchur(A,T)
			
				A:matUpper(R)
				A:pairwiseScaleAdd(-1, R, A)
			
				return A, nil
	-- 			return matEigenSystemFromSchur(A,T)
			end
			
			lastSum = thisSum
			
		end
		-- failed to converge
		return nil,nil
	end


	t.matInv    = matrix_invert
	t.matQR     = qr_decompose
	t.matEigen  = matEigenValues

	local help = MODTAB.help

	MODTAB.help = function(x)
		if x == matrix_invert then
			return
				"Get an element of the interaction tensor",
				"4 Integers, 1 String: destination layer (base 1), source layer (base 1), x and y offset, tensor name: XX, XY, XZ, etc",
				"1 Number: Tensor element"
		end
		if x == qr_decompose then
			return
				"Compute QR decomposition on a square matrix",
				"2 optional Arrays: destination for QR decomposition",
				"2 Arrays: QT decomposition"
		end
		if x == matEigenValues then
			return
				"Compute Eigen Values and perhaps Eigen Vectors",
				"1 optional number: Tolerance (default " .. _tol .. ")",
				"1 Array, 1 Array or nil: The diagonal of the 1st array contains the Eigen values, if the eigen array does not reduce to the Schur form then the second array has Eigen vectors in it's columns. In a future version Eigen vectors will be computed for Schur reduced matrices."
		end
		if x == nil then
			return help()
		end
		return help(x)
	end
end


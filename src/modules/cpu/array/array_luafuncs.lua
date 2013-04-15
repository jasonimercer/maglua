-- Array.Double

local MODNAME = "Array.Double"
local MODTAB = Array.Double
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

t.matInv    = matrix_invert


local help = MODTAB.help

MODTAB.help = function(x)
	if x == matrix_invert then
		return
			"Get an element of the interaction tensor",
			"4 Integers, 1 String: destination layer (base 1), source layer (base 1), x and y offset, tensor name: XX, XY, XZ, etc",
			"1 Number: Tensor element"
	end
	if x == nil then
		return help()
	end
	return help(x)
end

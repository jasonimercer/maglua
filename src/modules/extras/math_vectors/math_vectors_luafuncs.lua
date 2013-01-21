-- Math_Vectors

local function istable(a)
	return type(a) == "table"
end
local function isnumber(a)
	return type(a) == "number"
end
	
local function cross(a,b)
	if not istable(a) or not istable(b) then
		error("cross expects 2 tables")
	end
	
	local c = {}
	c[1] = a[2]*b[3] - a[3]*b[2]
	c[2] = a[3]*b[1] - a[1]*b[3]
	c[3] = a[1]*b[2] - a[2]*b[1]

	return c
end

local function dot(a,b)
	if not istable(a) or not istable(b) then
		error("dot expects 2 tables")
	end
	
	local sum = 0
	for i=1,table.maxn(a) do
		local x = a[i]
		local y = b[i]
		
		if isnumber(x) and isnumber(y) then
			sum = sum + x*y
		end
	end
	
	return sum
end

local function norm(a)
	return dot(a,a)^(1/2)
end
	
local function angleBetween(a,b)

	local n = norm(a) * norm(b)
	
	if n == 0 then
		return 0
	end
	
	local ct = dot(a, b) / n

	return math.acos(ct)
end

local function scaledVector(a, s)
	local r = {}
	for k,v in pairs(a) do
		r[k] = v*s
	end
	return r
end

local function project(a, b)
	local ab,bb = dot(a,b),dot(b,b)
	if bb == 0 then
		return scaledVector(b, 0)
	end
	return scaledVector(b, ab/bb)
end


local function rotateAboutBy(a, n, t)
	-- ortho vec
	local ux = n[1]
	local uy = n[2]
	local uz = n[3]

	-- normalize ortho vec
	local ulen = norm(n)
	
	if ulen > 0 then
		ux = ux / ulen
		uy = uy / ulen
		uz = uz / ulen
	else 
		ux, uy, uz = 1,0,0     
	end

	local sx = a[1]
	local sy = a[2]
	local sz = a[3]
	
	-- rotating based on Taylor, Camillo; Kriegman (1994). "Minimization on the Lie Group SO(3) and Related Manifolds". Technical Report. No. 9405 (Yale University).
	--
	-- 	Here is the rotation matrix:
	local cost = math.cos(t)
	local sint = math.sin(t)
	
	local R = {
		{ cost + ux^2 * (1-cost),   ux*uy*(1-cost) - uz*sint, ux*uz * (1-cost) + uy * sint},
		{ uy*ux*(1-cost) + uz*sint,   cost + uy^2 * (1-cost), uy*uz * (1-cost) - ux * sint},
		{ uz*ux*(1-cost) - uy*sint, uz*uy*(1-cost) + ux*sint, cost + uz^2 * (1-cost)} }
	
	-- now to multiply R * {sx,sy,sz}
	local rx = R[1][1]*sx + R[2][1]*sy + R[3][1]*sz
	local ry = R[1][2]*sx + R[2][2]*sy + R[3][2]*sz
	local rz = R[1][3]*sx + R[2][3]*sy + R[3][3]*sz
	
	return {rx,ry,rz}
end


math.cross = cross
math.dot = dot
math.rotateAboutBy = rotateAboutBy
math.angleBetween = angleBetween
math.norm = norm
math.scaledVector = scaledVector
math.project = project

local help = math.help or 
function(f)
	if f == nil then --want description for custom math scope
		return "This is the custom Math Scope for MagLua. The following are custom functions added to the base language to help " ..
			   "create and run simulations." --only returning 1 string
	end
end

math.help = function(x)
	if x == cross then
		return
			"Compute the cross product between 2 3-Vectors",
			"2 Tables of 3 Numbers: Input",
			"1 Table of 3 Numbers: Product"
	end
	if x == project then
		return
			"Compute the projection of one vector onto another",
			"2 Tables of Numbers: Input",
			"1 Table of Numbers: Projection"
	end
	if x == dot then
		return
			"Compute the dot product between 2 Vectors",
			"2 Tables of Numbers: Input",
			"1 Number: Product"
	end
	if x == norm then
		return
			"Compute the length of a vector",
			"1 Table of Numbers: Input",
			"1 Number: Length"
	end
	if x == angleBetween then
		return
			"Compute the angle between 2 vectors",
			"2 Tables of Numbers: Input",
			"1 Number: Angle"
	end
	if x == rotateAboutBy then
		return
			"Rotate a vector about another by a given numer of radians",
			"2 Tables of 3 Numbers, 1 Number: Source Vector, Rotation axis, radians to rotate",
			"1 Table of 3 Numbers: Rotated Vector"
	end	
	if x == scaledVector then
		return
			"Scale all elements in the given table by the numeric amount",
			"1 Table, 1 Number: Source Vector, Scale amount",
			"1 Table: Scaled Vector"
	end	
	
	
	
	
	if x == nil then
		return help()
	end
	return help(x)
end


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

	-- floating point problems:
	if ct < -1 then ct = -1 end
	if ct >  1 then ct =  1 end

	return math.acos(ct)
end

local function scaledVector(a, s)
	s = s or 1
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

local function randomNormal(mean, stddev)
	stddev = stddev or 1
	mean = mean or 0
	local x1, x2
	local w = 2
	while w >= 1 do
		x1 = math.random() * 2 - 1
		x2 = math.random() * 2 - 1
		w = x1*x1 + x2*x2
	end

	w = math.sqrt( (-2 * math.log(w)) / w)

	return stddev * x1 * w + mean;
end

local function colinear(v1, v2, tol)
	if v1 == nil or v2 == nil then
		return false
	end
	local tol = tol or 1e-12
	local ab = math.angleBetween(v1, {0,0,1})
	return (ab < tol) or (ab > math.pi-tol)
end

local function nullSpace(v1, v2, v3)
	if v1 == nil then -- no vectors
		return {1,0,0}, {0,1,0}, {0,0,1}
	end

	if v2 == nil then -- 1 vector
		if colinear(v1, {0,0,1}) then
			local q1 = math.scaledVector( math.cross( {1,0,0}, v1), 1)
			local q2 = math.scaledVector( math.cross( v1, q1), 1)
			return q1, q2
		end
		local q1 = math.scaledVector( math.cross( {0,0,1}, v1), 1)
		local q2 = math.scaledVector( math.cross( v1, q1), 1)
		return q1, q2
	end
	
	if v3 == nil then -- 2 vectors
		if colinear(v1, v2) then
			return math.nullSpace(v1, nil, nil)
		end

		return math.scaledVector(math.cross(v2, v1), 1)
	end
	
	-- 3 vectors
	if colinear(v1, v2) then
		return math.nullSpace(v1, v3, nil)
	end

	if colinear(v2, v3) then
		return math.nullSpace(v1, v3, nil)
	end

	if colinear(v1, v3) then
		return math.nullSpace(v2, v3, nil)
	end

	-- at this point the 3 input vectors span R3 so the basis of the nullspace is nil
	return
end

-- parse an agument list looking for a random object
-- return functions that call the uniform/normal methods
-- fall back on the math.random/math.randomNormal versions
local function get_random_un_funcs(arg)
	for k,r in pairs(arg) do
		local t = getmetatable(r) or {}
		local tu = t.uniform
		local tn = t.normal
		if (tu ~= nil) and (tn ~= nil) then
			local function u(a,b,c)
				return tu(r, a,b,c)
			end
			local function n(a,b,c)
				return tn(r, a,b,c)
			end
			return u,n
		end
	end
	return math.random, math.randomNormal
end

local function unwrap_table(t)
	if t[1] then
		local v = t[1]
		table.remove(t, 1)
		return v, unwrap_table(t)
	end
end
local function wrap_table(...)
	return arg
end

local function get_arg_type(arg, t, skip)
	skip = skip or 0
	local r = {}
	for i = skip+1, table.maxn(arg) do
		if type(arg[i]) == t then
			table.insert(r, arg[i])
		end
	end
	return unwrap_table(r)
end

local function get_numbers(arg, skip)
	return get_arg_type(arg, type(12), skip)
end
local function get_tables(arg, skip)
	return get_arg_type(arg, type({}), skip)
end
local function copy_table(t)
	local r = {}
	for k,v in pairs(t) do
		if type(v) == type({}) then
			r[k] = copy_table(v)
		else
			r[k] = v
		end
	end
	return r
end

-- rules:
-- table of rules. 
-- rule = {type(by character), probability of rule, tables, scalars}
-- tables define directions/plane, scalars define stddev
-- "A": axis with vector and stddev(0) 
-- "D": direction with vector and stddev(0)
-- "P": plane by 1 or 2 vectors and stddev(0)
-- "S": random on sphere
-- "Z": zero vector
local function vectorByRule(...)
	-- random functions
	local uf, nf = get_random_un_funcs(arg)
	
	local rules = copy_table( wrap_table( get_tables( arg )))[1]

	local total_prob = 0
	-- Now we need to format the rules in a more strict manner. 
	-- We will convert tables into 1 primary and 2 nullspace vectors
	for k,r in pairs(rules) do
		total_prob = total_prob + (r[2] or 0)
		local a, b, c = get_tables(r, 2)
		local stddev = get_numbers(r, 2) or 0
		if a == nil then
			a = {1,0,0}
		end
		if b == nil then
			b,c = math.nullSpace(a)
		else
			c = math.nullSpace(a,b)
			a,b,c = c,b,a
		end
		--          type     prob      space, stddev
		rules[k] = {r[1], (r[2] or 0), a,b,c, stddev}
	end
	local maxr = table.maxn(rules)

	return function()
		-- first pick a rule
		local uf, nf = get_random_un_funcs(arg)
		local dice = uf() * total_prob
		local rule = 1
		while dice > ( (rules[rule] or {})[2] or 0)  do
			if rules[rule] == nil then
				error("failed to pick a rule in math.vectorsByRule")
			end
			dice = dice - rules[rule][2]
			rule = rule + 1
		end

		rule = rules[rule]
		local a,b,c = rule[3], rule[4], rule[5]
		local stddev = rule[6]
		-- A D P S Z
		if rule[1] == "A" or rule[1] == "D" then -- axis/direction
			local x = math.rotateAboutBy(a, b, nf() * stddev)
			x = math.rotateAboutBy(x, a, uf() * math.pi * 2)
			if rule[1] == "A" and uf() > 0.5 then -- we can flip it
				x = {-x[1], -x[2], -x[3]}
			end
			return x
		end

		if rule[1] == "P" then
			local x = math.rotateAboutBy(b, c, nf() * stddev)
			x = math.rotateAboutBy(x, a, uf() * math.pi * 2)
			return x
		end

		if rule[1] == "S" then
			return math.scaledVector( {nf(), nf(), nf()}, 1)
		end

		if rule[1] == "Z" then
			return {0,0,0}
		end

		error("unknown rule: `" .. tostring(rule[1]) .. "'")

	end
	
end


local function vectorAboutAxis(_axis, _stddev)
	return math.vectorsByRules({{"A", 0, _axis, _stddev}})
end

local function vectorInPlane(a1, a2)
	return math.vectorsByRules({{"P", 0, a1, a2, 0}})
end


local function vectorAboutPlane(...)
	local rule = {"P", 0}
	for k,v in pairs(arg) do
		table.insert(rule, v)
	end
	return math.vectorsByRules({rule})
end

local function vectorAboutX(stddev)
	return math.vectorsByRules({{"D", 0, {1,0,0}, stddev}})
end

local function vectorAboutY(stddev)
	return math.vectorsByRules({{"D", 0, {0,1,0}, stddev}})
end

local function vectorAboutZ(stddev)
	return math.vectorsByRules({{"D", 0, {0,0,1}, stddev}})
end


local function vectorOnSphere()
	return math.vectorsByRules({{"S", 0}})
end



-- vector changed to vectors to help with
-- idea that multiple can be generated
math.vectorsOnSphere = vectorOnSphere
math.vectorsInPlane = vectorInPlane
math.vectorsAboutX = vectorAboutX
math.vectorsAboutY = vectorAboutY
math.vectorsAboutZ = vectorAboutZ
math.vectorsAboutAxis = vectorAboutAxis
math.vectorsAboutPlane = vectorAboutPlane
math.vectorsByRules = vectorByRule

math.nullSpace = nullSpace
math.randomNormal = randomNormal
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

	if x == vectorOnSphere then
		return
			"Generate a function that, when called, returns vectors on the unit sphere. Random numbers from the lua math.random.",
			"",
			"1 Function: When called, a table of 3 numbers is returned representing vectors on the unit sphere."
	end	

	if x == vectorAboutPlane then
		return 
			"Generate a function that, when called, returns random unit vectors near the plane defined by the input. Random numbers from the lua math.random.",
			"1 or 2 Tables of 3 Numbers, 1 Number: Normal of the plane or Basis vectors for the plane, standard deviation of angle out of the plane.",
			"1 Function: When called, a table of 3 numbers is returned representing vectors near the plane."
	end	

	if x == vectorInPlane then
		return
			"Generate a function that, when called, returns random unit vectors in the plane defined by the input. Random numbers from the lua math.random.",
			"1 or 2 Tables of 3 Numbers: Normal of the plane or Basis vectors for the plane.",
			"1 Function: When called, a table of 3 numbers is returned representing vectors in the plane."
	end	

	if x == vectorAboutAxis then
		return
			"Generate a function that, when called, returns random unit vectors rotated away from the input direction by an angle from a normal distribution with the given input stddev. Random numbers from the lua math.random.",
			"1 table of 3 numbers, 1 number: Primary axis, stddev of angle about that axis",
			"1 Function: When called, a table of 3 numbers is returned representing vectors about the given axis."
	end	


    if x == vectorAboutX then
		return "Syntactic sugar for math.vectorsAboutAxis({1,0,0}, stddev",
		       "1 Number: Standard deviation about X axis",
			   "1 Function: When called, a table of 3 numbers is returned representing vectors about the given axis."
	end
	
    if x == vectorAboutY then
		return "Syntactic sugar for math.vectorsAboutAxis({0,1,0}, stddev",
		       "1 Number: Standard deviation about Y axis",
			   "1 Function: When called, a table of 3 numbers is returned representing vectors about the given axis."
	end
	
    if x == vectorAboutZ then
		return "Syntactic sugar for math.vectorsAboutAxis({0,0,1}, stddev",
		       "1 Number: Standard deviation about Z axis",
			   "1 Function: When called, a table of 3 numbers is returned representing vectors about the given axis."
	end
	
	if x == randomNormal then
		return "Sample values from the normal distribution. The source of the values come from math.random().",
		       "0 or 1 or 2 Numbers: The optional values are mean and standard deviation with default values of 0 and 1.",
			   "1 Number: random value"
	end

	if x == math.erf then
		return "Compute the error function on an input",
		       "1 Number: input",
			   "1 Number: erf(input)"
	end
		
	if x == math.erfc then
		return "Compute the complimentary error function on an input",
		       "1 Number: input",
			   "1 Number: erfc(input)"
	end

	if x == math.nullSpace then
		return "Calculate the basis vectors of the null space of the input vector(s) in $\\mathbb{R}^3$",
		       "0, 1, 2 or 3 Tables of 3 Numbers: Input vectors that span some space in $\\mathbb{R}^3$",
	           "0, 1, 2 or 3 Tables of 3 Numbers: Basis vectors of the null space of the input in $\\mathbb{R}^3$"
	end
	
	if x == math.vectorsByRules then
		local example = [[<pre>
f = math.vectorsByRules({{"D", 1, {0,0,1}}, {"P", 2, {0,0,1}, 0.1}})
for i=1,10 do
    x = f()
    print(string.format("% 8f % 8f % 8f", x[1], x[2], x[3]))
end
</pre>]]
        local output = [[<pre> 0.952866 -0.299312 -0.049593
 0.987742 -0.113633 -0.107019
 0.000000  0.000000  1.000000
 0.502563 -0.864539  0.001873
 0.622044  0.781770  0.043541
 0.000000  0.000000  1.000000
 0.000000  0.000000  1.000000
 0.837640 -0.525051 -0.150601
 0.150530  0.978758 -0.139189
 0.980290 -0.123300  0.154365</pre>
						]]
 return "Create a function that returns vectors according to defined rules and probabilities. Example usage:" .. example .. "The above creates a function that, when called, returns a vector in the {0,0,1} direction 1/3 of the time and a vector mostly in the XY plane (with a rotation out of the plane governed by a normal distribution with a standard deviation of 0.1) 2/3 of the time. Sample output:" .. output,
		[[Table of rules, Optional RNG object: A rule is a table of {rule type identifier, rule probability, 0, 1 or 2 tables of 3 numbers representing axes and an optional number used for standard deviations}. The rule identifiers are the single upper case characters "A", "D", "P", "S", "Z" which are axis, direction, plane, sphere, zero vector. The probabilities of the rules will be summed and normalized so one can be 18 and the other can be 2 reulting in probabilities of 0.9 and 0.1. The tables of 3 numbers define either directions, axes, planes or plane normals.]],
		"1 Function: Each time the returned function is called it will return a table of 3 numbers. If a RNG object was supplied in the initial call, the values will be based on that random stream otherwise the built-in math.random will be used."
	end

	
	if x == nil then
		return help()
	end
	return help(x)
end


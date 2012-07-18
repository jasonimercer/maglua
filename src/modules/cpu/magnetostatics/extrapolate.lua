tol = 1e-8

-- assuming t is the {x,y} values of something that converges slowly
-- 
-- idea is to take log of {x,y}' and see if the resulting points fit
-- a straight line, if so, extrapolate, else, return false
function extrapolate(t, mult) --to infinity, mult for flipping y (can only have positive derivs)
-- 	for k,v in pairs(t) do
-- 		print(k,v[1], v[2])
-- 	end
	
	while table.maxn(t) > 6 do
		table.remove(t, 1)
	end
	
	if mult then
		for k,v in pairs(t) do
			t[k][2] = mult*v[2]
		end
	end
	local N = table.maxn(t)
	local lastx = t[N][1]
	local lasty = t[N][2]
	
	if lasty == 0 then
		return 0
	end
	
	local d = deriv(t)
	for k,v in pairs(d) do
		if v[2] < 0 and mult == nil then 
			return extrapolate(t, -1)
		end
	end
	
	for k,v in pairs(d) do
		d[k] = {math.log(v[1]), math.log(v[2])}
	end
	
	local N = table.maxn(d)
	local m = {}
	local b = {}
	
	-- get each pairwise eqn of line
	for i=1,N-1 do
		local mi, bi = line(d[i], d[i+1])
		m[i] = mi
		b[i] = bi
	end
	
	-- mean, stddev to see how straight the data is
	local mu, ms = stats(m)
	local bu, bs = stats(b)
	
	if ms > tol or bs > tol then
		return false
	end
	
	-- we now have a good equation of a line in log-log space for the
	-- derivative of the data. Lets write the function in linear space	
	-- ln(y) = m ln(x) + b
    -- exp(ln(y)) = exp(m ln(x) + b)
	--        y   = exp(b) x^m
	--
	-- finally, this is the derivative of the data, so lets integrate it
	--       Y    = integral(e^b x^m, dx)
	--       Y    = e^b x^(m+1) / (m+1)
	-- 
	-- the Limit of this function as it goes to infinity is zero so the
	-- value at the lastx is what needs to be added onto lasty to get
	-- an approximation of the value at infinity

	local m, b = m[N-1], b[N-1] -- last pair are best 
	local function f(x)
		return math.exp(b) * x^(m+1) / (m+1)
	end
	
	local solution = lasty + (0 - f(lastx))
	
	if mult ~= nil then
		solution = mult * solution 
	end
	return solution
end

function stats(t)
	local u, s = 0, 0
	local N = table.maxn(t)
	
	for i=1,N do
		u = u + t[i]
	end
	u = u / N
	
	for i=1,N do
		s = s + (t[i] - u)^2
	end
	s = s / N
	return u,s
end

function deriv(t)
	local d = {}
	
	local N = table.maxn(t)
	
	for i=2,N-1 do
		local m1, m2
		m1 = (t[i][2] - t[i-1][2])/(t[i][1] - t[i-1][1])
		m2 = (t[i+1][2] - t[i][2])/(t[i+1][1] - t[i][1])
		table.insert(d, {t[i][1], (m1+m2)/2})
	end
	return d
end

function line(p1, p2)
	--y = m x + b
	local m, b
	m = (p1[2] - p2[2]) / (p1[1] - p2[1])
	local x, y = p1[1], p1[2]
	
	b = y - m * x
	return m, b
end

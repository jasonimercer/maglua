-- This script makes the FFT plan

-- attempt to factor into small primes
function factor(v)
	local primes = {4,2,3,5,7,11,13,17}
	--local primes = {17,13,11,7,5,3,2}
	local f = {}
	
	for i=1,7 do
		local d = primes[i]
		while math.mod(v, d) == 0 do
			table.insert(f, d)
			v = v / d
		end
	end
	
	if v ~= 1 then
		error("Failed to factor `" .. v .. "'")
	end
	return f
end

-- split table into R subtables
function split(t, R)
	local T = {}
	for i=1,R do
		T[i] = {}
	end
	for i,v in pairs(t) do
		table.insert(T[math.mod(i-1,R)+1], v)
	end
	return T
end

function fft_plan(indices, in_factors, depth)
	local depth = depth or 1
	local R = in_factors[1]
	local ff = {}
	for i=2,table.maxn(in_factors) do
		ff[i-1] = in_factors[i]
	end
	
	local N = table.maxn(indices)
	if N == 1 then
		return indices
	end
	local T = split(indices, R)
	
	
	for i=1,R do
		T[i] = fft_plan(T[i], ff, depth+1)
	end
	
	for k=1,N do
		local j = ((k-1)%(N/R))+1
		local src = {}
		for q=1,R do
			table.insert(src, {numerator=k-1, denominator=N, power=q-1, src=T[q][j], radix=R})
		end
		
		table.insert(plan, {depth=depth, dest=indices[k], src=src})		
	end
	
	return indices
end

plan = {}
indices = indices or {1,2,3,4,5,6,7,8}
factors = factor(table.maxn(indices))
fft_plan(indices, factors)
max_depth = table.maxn(factors)


-- What do we have now?
-- 
-- plan[integer] = {depth, dest, src[]}
--     src[integer] = {numerator, denominator, power, src, radix}
--
-- lets move the info in src into the main table

plan2 = {}
for k,v in pairs(plan) do
	local s = {}
	local p = {}

	for a,b in pairs(v.src) do
		table.insert(s, b.src)
		table.insert(p, b.power)
	end

	table.insert(plan2, {depth=v.depth, dest=v.dest, radix=v.src[1].radix, numerator=v.src[1].numerator, denominator=v.src[1].denominator, src=s, power=p})
end

-- divide up into depths
plan3 = {}
for k,v in pairs(plan2) do
	if plan3[v.depth] == nil then
		plan3[v.depth] = {}
	end

	table.insert(plan3[v.depth], v)
end

function same_tables(a,b,s)
	for k,v in pairs(a) do
		if a[k] ~= b[k] then
			return false
		end
	end
	if s == nil then
		return same_tables(b,a,1)
	end
	return true
end


-- divide up into common sources
plan4 = {}
for d=1,max_depth do
	plan4[d] = {}
	for k,v in pairs(plan3[d]) do
		local found = false

		for x,y in pairs(plan4[d]) do
			if same_tables(x, v.src) then
				found = true
				table.insert(plan4[d][x], v)
			end
		end

		if not found then
			plan4[d][v.src] = {v}
		end
	end
end

-- sources are keys, lets make them fields
plan5 = {}
for d=1,max_depth do
	plan5[d] = {}
	for k,v in pairs(plan4[d]) do
		local t = v
		t.srcs = k
		table.insert(plan5[d], t)
	end
end

-- finally lets sort by sources
function src_srt(a, b)
	return a.srcs[1] < b.srcs[1]
end
for d=1,max_depth do
	table.sort(plan5[d], src_srt)
end


function print_common_plan()
	for d,pp in pairs(plan5) do
		for k,p in pairs(pp) do
			print(d, "srcs: " .. table.concat(p.srcs, ","))
			for i=1,table.maxn(p) do
				print("", p[i].dest, p[i].numerator .. "/" .. p[i].denominator, table.concat(p[i].power, ","))
			end
		end
	end
end



function table_copy(t)
	local copy = {}
	for k,v in pairs(t) do
		copy[k] = v
	end
	return copy
end

function table_expand(t)
	local i = t[1]
	table.remove(t, 1)
	if t[1] then
		return i, table_expand(t)
	end
	return i
end

function get_num_plan(depth)
	return table.maxn(plan5[depth])
end

function get_dest_count(depth)
	return table.maxn(plan5[depth][1])
end

function get_radix(depth)
	return plan5[depth][1][1].radix
end

function get_plan(depth, i)
	local p = plan5[depth][i]

	if p == nil then
		error("Failed to find plan details for depth = " .. depth .. ", index = " .. i)
	end
	
	local c = table_copy(p.srcs)
	local f = {}
	local d = {}
	for i in ipairs(p) do
		table.insert(c, p[i].numerator/p[i].denominator)
	end
	for i in ipairs(p) do
		table.insert(c, p[i].dest)
	end
	
-- 	print(depth, i,  table.concat(c, ","))
	--return srcs, fracs, dests
	return table_expand(c)
end

-- print(get_radix(1))
-- print(get_plan(1,1))


-- print(get_plan(10,3))
-- print("New plan")
-- print_plan(plan)
-- print()
-- print_common_plan()

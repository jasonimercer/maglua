-- FitEnergy
local MODNAME = "FitEnergy"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time
local methods = {}


-- internal support functions
local function get_fe_data(fe)
    if fe == nil then
        return {}
    end
    if fe:getInternalData() == nil then
        fe:setInternalData({})
    end
    return fe:getInternalData()
end

local function ct(t) -- copy table
    local x = {}
    for k,v in pairs(t) do
        x[k] = v
    end
    return x
end

methods["addData"] =
{
"Add data to solve the fit:",
"1 Table of spin configurations, 1 Number: The spin configurations can be VectorCS objects or a list compatible with it's format. The number is the resulting energy with that configuration used.",
"",
function(fe, cfgs, energy)
    local d = get_fe_data(fe)
    d.samples = d.samples or {}

    local cc = {}
    for k,v in pairs(cfgs) do
        cc[k] = VectorCS.new(v)
    end

    table.insert(d.samples, {cc, energy})
end
}

local function get_term(pat, dat)
    local vars = {}

    for k,v in pairs(dat) do
        vars[k] = VectorCS.new(v)
        vars[k]:setMagnitude(1)
    end

    if table.maxn(pat) == 4 then
        local a = pat[1]
        local b = pat[2]
        local i = pat[3]
        local j = pat[4]
        
        return vars[i]:toTable()[a] * vars[j]:toTable()[b]
    end

    if table.maxn(pat) == 2 then
        local a = pat[1]
        local i = pat[2]
        
        return vars[i]:toTable()[a]
    end

    if table.maxn(pat) == 1 then
        return 1
    end
    
    error("bad pattern")
end

local function populate_row(A, r, terms, data)
    for c=1,A:nx() do
        local a = get_term(terms[c], data)
        A:set(c,r, a)
    end
end


methods["calculate"] =
{
"Use the solved fitting relationship to calculate the energy associated with the provided spin configuration.",
"1 Table of spin configurations: The spin configurations can be VectorCS objects or a list compatible with it's format.",
"1 Number: Resulting energy",
function(fe, cfg)
    local data = get_fe_data(fe)

    if data.A == nil then
        error("need to fit data first")
    end

    populate_row(data.A, 1, data.terms, cfg)

    data.A:matMul(data.x, data.b)

    return data.b:get(1)
end
}

methods["fitData"] =
{
"",
"",
"",
function(fe)
    local data = get_fe_data(fe)
    local samples = data.samples or {}

    -- We will couple all the components of all the spins with Gamma^{\alpha \beta}_{i j} terms.
    -- 
    -- Also adding in a scaling terms Gamma^{\alpha}_{i}
    --
    -- And having a consant term
    -- 
    -- All the spins will be scaled to unity
    
    local terms = {}

    -- couple all the components of the spins together
    -- and couple the spins with a constant external spin
    for i=1,2 do
        for j=i,2 do
            for a=1,3 do
                for b=a,3 do
                    table.insert(terms, {a,b,i,j})
                end
            end
        end
    end
    for i=1,2 do
        for a=1,3 do
            table.insert(terms, {a,i})
        end
    end
    table.insert(terms, {0})

    -- adding enough rows to compensate for the relationships between X, Y and Z
    -- from the fact that they are on the surface of a sphere
    local extra_rows = 0
    local n = table.maxn(terms)
    local rows = n + extra_rows
    if table.maxn(samples) < rows+1 then -- +1 for the constant term
        return false, "Need at least " .. rows+1 .. " samples (got " .. table.maxn(samples) .. ")"
    end

    local A = Array.Double.new(n, rows)
    local b = Array.Double.new(1, rows)

    for r=1,rows do
        populate_row(A, r, terms, samples[r][1])
        b:set(1,r, samples[r][2])
    end

    local cc = {"X", "Y", "Z"}
    local term_names = {}
    for k,v in pairs(terms) do
        if table.maxn(v) == 1 then
            term_names[k] = 1
        end
        if table.maxn(v) == 2 then
            term_names[k] = cc[v[1]] .. v[2]
        end
        if table.maxn(v) == 4 then
            term_names[k] = cc[v[1]] .. cc[v[2]] .. v[3] .. v[4]
        end
    end

    local x,msg,method = A:matLinearSystem(b)

    if x == nil then
        return false, msg
    end

    data.terms = terms
    data.x = x

    -- workspaces:
    data.A = Array.Double.new(A:nx(), 1)
    data.b = Array.Double.new(1, 1)

    -- telling the C side about the fit
    fe:clearTerms()
    for k,t in pairs(terms) do
        if table.maxn(t) == 4 then
            fe:addTerm(t[1], t[2], t[3], t[4])
        end
        if table.maxn(t) == 2 then
            fe:addTerm(t[1],   -1, t[2],   -1)
        end
        if table.maxn(t) == 1 then
            fe:addTerm(  -1,   -1,   -1,   -1)
        end
    end
    
    fe:setX( x:matTrans():toTable(1) )

    return true
end
}




-- inject above into existing metatable for FitEnergy
for k,v in pairs(methods) do
    t[k] = v[4]
end

-- backup old help function for fallback
local help = MODTAB.help

-- create new help function for methods above
MODTAB.help = function(x)
		  for k,v in pairs(methods) do
		      if x == v[4] then
			  return v[1], v[2], v[3]
		      end
		  end

		  -- fallback to old help if a case is not handled
		  if x == nil then -- FitEnergy overview
		      return [[Fit energy function to an analytic form]], "", ""
		  end
		  return help(x)
	      end



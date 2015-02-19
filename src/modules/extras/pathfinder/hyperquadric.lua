-- This script generalizes quadrics to arbitrary dimensions.
-- The goal is to fit data to them to find critical points

-- let's write some lower dimension functions to get the feel for them
-- 1D
-- value = a x^2 + b x + c
--
-- 2D
-- value = a x^2 + b y^2 + c x y + d x + e y + g
--
-- 3D
-- value = a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z + j

-- now let's generate these via code
local function makeEquation(dims)
    -- we will store each variable by an integer
    -- zero will denote the constant term 1
    local eqn = {}
    
    for i=1,dims do
        table.insert(eqn, {i,i}) -- terms like a x^2
    end

    for i=1,dims-1 do
        for j=i+1,dims do
            table.insert(eqn, {i,j}) -- terms like b x y
        end
    end

    for i=1,dims do
        table.insert(eqn, {i, 0}) -- terms like c x
    end

    table.insert(eqn, {0, 0}) -- final constant term
    return eqn
end

local function Deqn(eqn, var)
    -- Deqn will do this:
    -- input:  Deqn({{1,1}, {1,2}, {2,2}, {1, 0}}, 1)
    -- output:      {{2, {1}}, {1, {2}}, {0, {2,2}}, {1, {0}} }

    -- count & remove 1 in table
    -- this is used to differentiate {1,1} (x^2)  to {2, {1}} (2 x)
    local function cr(t,x) 
        local c = 0
        local r = {}
        for k,v in pairs(t) do
            if v == x then
                c = c + 1
                if c ~= 1 then
                    table.insert(r, v)
                end
            else
                table.insert(r, v)
            end
        end
        return {c, r}
    end

    local deqn = {}
    for i=1,table.maxn(eqn) do
        deqn[i] = cr(eqn[i], var)
    end
    return deqn
end

local function add_coef_to_deqn(x, deqn)
    for i=1,table.maxn(deqn) do
        deqn[i][1] = deqn[i][1] * x:get(1,i)
    end
    return deqn
end

-- combine terms, order based on variable number
local function simplify_deqn(deqn, dims)
    local simp = {}
    for d=0,dims do
        simp[d] = 0
        for i=1,table.maxn(deqn) do
            if deqn[i][2][1] == d then
                simp[d] = simp[d] + deqn[i][1]
            end
        end
    end
    return simp
end


local function hq_build_M_matrix(data)
    local dims = table.maxn(data[1]) - 1
    local eqn = makeEquation(dims)

    local n = table.maxn(eqn)
    if table.maxn(data) < n then
        return nil, "Need more data"
    end

    local M = Array.Double.new(n,n)
    local b = Array.Double.new(1,n)

    for x=1,n do
        for y=1,n do
            local v = 1
            for k=1,table.maxn(eqn[x]) do
                local var = eqn[x][k]
                v = v * (data[y][var] or 1)
            end
            M:set(x,y,1, v)
        end
    end

    return M
end

local function hq_build_b_vector(data)
    local dims = table.maxn(data[1]) - 1
    local eqn = makeEquation(dims)

    local n = table.maxn(eqn)
    if table.maxn(data) < n then
        return nil, "Need more data"
    end

    local b = Array.Double.new(1,n)

    for y=1,n do
        b:set(1,y,1, data[y][dims+1])
    end

    return b
end


-- now we will solve for the coefficients given data
-- and then solve for the critical points using those
-- coefficients and the derivatives of the equation
local function hq_solve_critical(data, M, b)
    if data == nil or data[1] == nil then
        return nil, "Need more data"
    end

    local dims = table.maxn(data[1]) - 1
    local eqn = makeEquation(dims)

    local n = table.maxn(eqn)
    if table.maxn(data) < n then
        return nil, "Need more data"
    end

    local x = M:matLinearSystem(b)
    --[[
    M:matPrint("M1", {mathematica=true})
    local x = M:matInv():matMul(b)
    M:matPrint("M2", {mathematica=true})
    b:matPrint("b", {mathematica=true})
    x:matPrint("x", {mathematica=true})

    print("mat cond", M:matCond())
    --]]

    -- now we have the coeficients of the equation 
    -- we can solve for the derivatives = 0

    local function print_eqn(e)
        local t = {}
        for k,v in pairs(e) do
            table.insert(t, "("..table.concat(v, " ") .. ")")
        end
        print(table.concat(t, " + "))
        print()
    end

    local function print_deqn(e,c)
        local t = {}
        for k,v in pairs(e) do
            table.insert(t, string.format("(%g (%s))", v[1], table.concat(v[2], " ")))
        end
        print("d/d(" .. c .. "):",table.concat(t, " + "))
        print()
    end

    local B = Array.Double.new(dims, dims)
    local c = Array.Double.new(   1, dims)
    for d=1,dims do
        -- print_eqn(eqn)

        -- differentiate wrt the dimension
        deqn_d = Deqn(eqn, d)
        -- print_deqn(deqn_d, d)

        -- multiply in the coefficients
        deqn_d = add_coef_to_deqn(x, deqn_d)
        -- print_deqn(deqn_d, d)
 
        -- combine like terms
        deqn_d = simplify_deqn(deqn_d, dims)
        -- print(table.concat(deqn_d, ", "))

        --interactive()
        -- build the matrix
        for i=1,dims do
            B:set(i,d,1,  deqn_d[i])
        end

        -- build the vector
        -- the deqn_d[0] is the constant term, we negate it since it's 
        -- moving to the other side of the equals sign
        c:set(1,d,1, -deqn_d[0])
    end

    --local opts = {post="", format="% 4.3f", delim="  "}
    --B:matPrint("B", opts)
    --c:matTrans():matPrint("c^T", opts)

    local z = B:matLinearSystem(c)
    -- local z = B:matInv():matMul(c)
    
    -- z:matTrans():matPrint("z^T", opts)
    -- interactive()

    return z:matTrans():toTable(1)
end


local function sampleAbout(base, coords, amount)
    local function g() return 2 * (math.random() - 1) end
    local function r(x) return x*g() end
    
    local p = {}
    for k,v in pairs(base) do
        p[k] = v
    end
    
    for k,c in pairs(coords) do
        p[c] = p[c] + r(amount[k])
    end
    
    return p
end

local function hq_builddata(guess, terms, amount, n, value_func)
    -- create data around guess:
    local data = {}
    for i=1,n do
        local s = sampleAbout(guess, terms, amount)
        data[i] = {}
        for j=1,table.maxn(terms) do
            table.insert(data[i], s[terms[j]])
        end
        table.insert(data[i], value_func(s))
    end
    return data
end

function hq_refine(guess, terms, amount, value_func)
    local dims = table.maxn(terms)
    local n = table.maxn(makeEquation(dims))

    -- make a copy of the guess
    -- this will get updated with refined 
    -- terms at the end
    local g2 = {}
    for k,v in pairs(guess) do
        g2[k] = v
    end
    

    local data = hq_builddata(guess, terms, amount, n, value_func)
    local M = hq_build_M_matrix(data)
    local b = hq_build_b_vector(data)

    --[[
    M:matPrint("M", {mathematica=true})
    b:matPrint("b", {mathematica=true})
    local x = M:matLinearSystem(b)
    x:matPrint("x", {mathematica=true})
    --]]

    local s = hq_solve_critical(data, M, b)

    -- insert updated terms back into guess and return
    for i=1,table.maxn(terms) do
        g2[terms[i]] = s[i]
    end
    return g2
end




dofile("hyperquadric.lua")

ss = SpinSystem.new(1,1,2)
ex = Exchange.new(ss)
ani = Anisotropy.new(ss)
zee = AppliedField.new(ss)

J = 3
K = 1
m = 1

s1 = {1,1,1}
s2 = {1,1,2}

ss:setSpin(s1, {0,0,1}, m)
ss:setSpin(s1, {0,0,1}, m)

ex:add(s1, s2, J)
ex:add(s1, s2, J)

ani:add(s1, {0,0,1}, K)
ani:add(s2, {0,0,1}, K)

zee:set(1, 0, 0.1)

function s2c(t,p,r)
    if type(t) == type({}) then
        local a,b,c = s2c(t[1], t[2], t[3])
        return {a,b,c}
    end
    return 
    math.cos(p)*math.sin(t)*r,
    math.sin(p)*math.sin(t)*r,
    math.cos(t)*r
end

function c2s(x,y,z)
    if type(x) == type({}) then
        local a,b,c = c2s(x[1], x[2], x[3])
        return {a,b,c}
    end
    local r = (x^2 + y^2 + z^2)^(1/2)
    return math.acos(z/r), math.atan(y,x), r
end

-- This energy function isn't correct. We don't have the 
-- 1/2 terms but it will generate good test data
function energyCartesian(p1, p2)
    ss:resetFields()

    ss:setSpin(s1, p1, m)
    ss:setSpin(s2, p2, m)
    
    ex:apply(ss)
    ani:apply(ss)
    zee:apply(ss)

    ss:sumFields()
    
    local e = 0

    for z=1,2 do
        local sx, sy, sz = ss:spin({1,1,z})
        local hx, hy, hz = ss:field("Total", {1,1,z})

        e = e - (sx*hx + sy*hy + sz*hz)
    end

    return e
end


function energySpherical(t1, p1, t2, p2)
    local x1,y1,z1 = s2c(t1,p1,m)
    local x2,y2,z2 = s2c(t1,p1,m)

    return energyCartesian({x1,y1,z1}, {x2,y2,z2})
end


pf = PathFinder.new()

pf:setEnergyFunction(energyCartesian)

    upup = {{0,0, m}, {0,0, m}}
downdown = {{0,0,-m}, {0,0,-m}}

path = pf:findPath(upup, downdown, 2)


max_idx = 1
max_en = energyCartesian(path[1][1], path[1][2])

for k,v in pairs(path) do
    local e = energyCartesian(v[1], v[2])

    if e > max_en then
        max_idx = k
        max_en = e
    end
end

saddle_point = path[max_idx]

print(table.concat(saddle_point[1], "\t"))
print(table.concat(saddle_point[2], "\t"))

print(table.concat(c2s(saddle_point[1]), "\t"))
print(table.concat(c2s(saddle_point[2]), "\t"))




-- We now have a function of 4 variables that's quadric-like around 
-- the saddle point. Let's give our solver 14 data points around 
-- the saddle point and see if it refines


function improve_guess(g1,g2,g3,g4)
    local hq = HyperQuadric.new()

    guess = {g1, g2, g3, g4}

    for i=1,15 do
        local t1 = math.random()*2*math.pi
        local t2 = math.random()*2*math.pi
        local g1 = guess[1] + 0.1 * (math.random()-0.5)
        local g2 = guess[2] + 0.1 * (math.random()-0.5)
        local g3 = guess[3] + 0.1 * (math.random()-0.5)
        local g4 = guess[4] + 0.1 * (math.random()-0.5)
        
        local en = energySpherical(g1,g2,g3,g4)

        -- print(g1,g2,g3,g4,en)

        hq:addData(g1,g2,g3,g4, en)
    end

    a = hq:solveCriticalPoint()

    g1 = a:get(1,1)
    g2 = a:get(1,2)
    g3 = a:get(1,3)
    g4 = a:get(1,4)

    return g1,g2,g3,g4
end


t = c2s(saddle_point[1])
g1,g2 = t[1], t[2]
t = c2s(saddle_point[2])
g3,g4 = t[1], t[2]


--g1,g2,g3,g4 = 0.1,0,0.03,0

print(string.format("% 10g % 10g % 10g % 10g", g1,g2,g3,g4))

for i=1,40 do
    g1,g2,g3,g4 = improve_guess(g1,g2,g3,g4)
    print(string.format("% 10g % 10g % 10g % 10g", g1,g2,g3,g4))
end


-- This script solves for the critical point of the following equation
--
-- Equation:
--    a x^2 + 2 b x y + 2 c x z + 2 d x w 
--  + e y^2 + 2 f y z + 2 g y w 
--  + h z^2 + 2 i z w 
--  + j w^2 
--  + k x + l y + m z + n w + p = energy
--
-- Partial Derivatives:
--   [a b c d] [x]   [k]
-- 2 [b e f g] [y] =-[l]
--   [c f h i] [z]   [m]
--   [d g i j] [w]   [n]
--
-- This is used to refine critical points for functons of 4 variables
-- that have quadratic features

local function hq_solveCriticalPoint(self)
    if table.maxn(self.data) < 15 then
        return nil, "Need more data"
    end

    local data = self.data
    local A = Array.Double.new(15,15)
    local b = Array.Double.new( 1,15)    

    for i=1,15 do
        local x,y,z,w = data[i][1],data[i][2],data[i][3],data[i][4]

        local t = {x*x, 2*x*y, 2*x*z, 2*x*w, y*y, 2*y*z, 2*y*w, z*z, 2*z*w, w*w, x, y, z, w, 1}

        for j=1,15 do
            A:set(j,i,1,  t[j])
        end

        b:set(1,i,1,  data[i][5])
    end

    local x, msg = A:matLinearSystem(b)

    -- solving for critical point
    -- D y = e
    local D = Array.Double.new(4,4)

    local lookup = {a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8,i=9,j=10,k=11,l=12,m=13,n=14}

    D:set(1,1,1, x:get(1,lookup.a))
    D:set(2,1,1, x:get(1,lookup.b))
    D:set(3,1,1, x:get(1,lookup.c))
    D:set(4,1,1, x:get(1,lookup.d))

    D:set(1,2,1, x:get(1,lookup.b))
    D:set(2,2,1, x:get(1,lookup.e))
    D:set(3,2,1, x:get(1,lookup.f))
    D:set(4,2,1, x:get(1,lookup.g))

    D:set(1,3,1, x:get(1,lookup.c))
    D:set(2,3,1, x:get(1,lookup.f))
    D:set(3,3,1, x:get(1,lookup.h))
    D:set(4,3,1, x:get(1,lookup.i))

    D:set(1,4,1, x:get(1,lookup.d))
    D:set(2,4,1, x:get(1,lookup.g))
    D:set(3,4,1, x:get(1,lookup.i))
    D:set(4,4,1, x:get(1,lookup.j))


    local e = Array.Double.new(1,4)
    e:set(1,1,1, -0.5 * x:get(1,lookup.k))
    e:set(1,2,1, -0.5 * x:get(1,lookup.l))
    e:set(1,3,1, -0.5 * x:get(1,lookup.m))
    e:set(1,4,1, -0.5 * x:get(1,lookup.n))

    local y = D:matLinearSystem(e)
    interactive()

    return y
end

HyperQuadric = {}

HyperQuadric.new = 
function()
    local hq = {}

    hq.data = {}

    hq.addData = function(self, x,y,z,w, f)
                     table.insert(self.data, {x,y,z,w, f})
                 end

    hq.solveCriticalPoint = hq_solveCriticalPoint

    return hq
end


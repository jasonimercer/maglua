-- This file provides the function `ColorMap(x,y,z)' that converts a direction (x,y,z) into red, green, blue
-- This file provides the table `ColorMap' which has the following keys:
-- 
-- "HSVtoRGB"
-- "HSLtoRGB"
--
-- Example Usage:
-- <pre>
-- dofile("maglua://ColorMap.lua")
--
-- r, g, b = ColorMap.HSVtoRGB(1, 0, 0.5)
-- </pre>


ColorMap = {}

local function Hprime2RGB1(Hprime, C, X)
    local R1, G1, B1 = 0,0,0

    if Hprime >=0 and Hprime < 1 then
        R1, G1, B1 = C, X, 0
    end
    if Hprime >=1 and Hprime < 2 then
        R1, G1, B1 = X, C, 0
    end
    if Hprime >=2 and Hprime < 3 then
        R1, G1, B1 = 0, C, X
    end
    if Hprime >=3 and Hprime < 4 then
        R1, G1, B1 = 0, X, C
    end
    if Hprime >=4 and Hprime < 5 then
        R1, G1, B1 = X, 0, C
    end
    if Hprime >=5 and Hprime <=6 then
        R1, G1, B1 = C, 0, X
    end
    return R1, G1, B1
end

ColorMap["HSVtoRGB"] = 
function(H,S,V)
    local C = V * S
    
    while H > 360 do
        H = H - 360
    end
    while H < 0 do
        H = H + 360
    end

    local Hprime = H / 60

    local X = C * (1 - math.abs(math.mod(Hprime, 2) - 1))

    local R1, G1, B1 = Hprime2RGB1(Hprime, C, X)

    local m = V - C
    return R1+m, G1+m, B1+m
end

ColorMap["HSLtoRGB"] = 
function(H,S,L)
    local C = (1 - math.abs(2*L - 1)) * S
    
    while H > 360 do
        H = H - 360
    end
    while H < 0 do
        H = H + 360
    end

    local Hprime = H / 60

    local X = C * (1 - math.abs(math.mod(Hprime, 2) - 1))

    local R1, G1, B1 = Hprime2RGB1(Hprime, C, X)

    local m = L - 0.5*C
    return R1+m, G1+m, B1+m
end



--[[

function ColorMap(x, y, z)

	local r = (x*x+y*y+z*z)^(1/2)
	
	if r == 0 then
		return 0,0,0
	end
	local theta = math.atan2(y, x)
	local phi   = math.acos(z/r)

	local pi = math.pi
	local cos = math.cos
	local sin = math.sin

	local r, g, b = 1, 1, 1

	-- theta [-pi:pi]
	if theta < 0 then
		theta = theta + 2*pi
	end
	-- theta [0:2pi]
	
	if phi < 0.5 * pi then
		if theta < 0.5 * pi then
			r = (-0.5 * cos(2.0 * phi) + 0.5) 
			g = 1.0 - ((0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
			b = (0.5 * cos(2.0 * phi) + 0.5)
		else if theta < pi then
			r = (-0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5)
			g = 1.0
			b = (0.5 * cos(2.0 * phi) + 0.5)
		else if theta < 1.5 * pi then
			r = (0.0) * (-cos(2.0 * phi) + 0.5)
			g = 1.0 - ((-0.5 * cos(2.0 * phi) + 0.5) * (-0.5 * cos(2.0 * theta) + 0.5))
			b = 1.0 - ((0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
		else
			r = (0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5)
			g = (0.5 * cos(2.0 * phi) + 0.5)
			b = 1.0 - ((0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
		end
		end
		end
	else
		if theta < 0.5 * pi then
			r = 1.0
			g = ((-0.5 * cos(2.0 * theta) + 0.5 ) * (-0.5 * cos(2.0 * phi) + 0.5))
			b = 0.5 * cos(2.0 * phi) + 0.5
		else if theta < pi then
			r = 1.0 - ((0.5 * cos(2.0*theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
			g = (-0.5 * cos(2.0 * phi) + 0.5)
			b = 0.5 * cos(2.0 * phi) + 0.5
		else if theta < 1.5 * pi then
			r = 0.5 * cos(2.0 * phi) + 0.5
			g = ((0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
			b = 1.0 - ((0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
		else
			r =  1.0 - ((-0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
			g =  0.0
			b =  1.0 - ((0.5 * cos(2.0 * theta) + 0.5) * (-0.5 * cos(2.0 * phi) + 0.5))
		end end end
	end

	return r, g, b
    end
--]]
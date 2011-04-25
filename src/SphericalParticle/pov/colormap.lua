function colormap(x, y, z)
-- 	z = -z
-- 	if z < 0 then
-- 		r = 1
-- 		g = 1+z
-- 		b = 1+z
-- 	else
-- 		r = 1-z
-- 		g = 1-z
-- 		b = 1
-- 	end
-- 	
-- 	x = 0.9
-- 	y = 1.0 - x
-- 	r, b, g = r*x+y, b*x+y, g*x+y
-- 	
-- 	if true then
-- 		return r,g,b
-- 	end
	
	theta = math.atan2(y, x)
	phi   = math.acos(z)

	local pi = math.pi
	local cos = math.cos
	local sin = math.sin

	r, g, b = 1, 1, 1

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

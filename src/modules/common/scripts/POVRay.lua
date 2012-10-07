-- this file generates a povray rendering of a spinsystem
-- the function to call and arguments are:
-- 
--  pov(filename, SpinSystem, {camx, camy, camz}, {cam_atx, cam_aty, cam_atz}, 
--     {{light1x, light1y, light1z}, {light2x, light2y, light2z}, {light3...}, ...}},  
--      optional_colour_func )
-- 
--  The colour function takes an x, y, z and returns r, g, b

local function colormap(x, y, z)

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


local function _lights(tab)
	s = ""
	for k,v in pairs(tab) do
		s = s .. string.format(
				"light_source {\n" ..
				"   <%f,%f,%f>, rgb <1, 1, 1>\n" ..
				"   fade_distance 100\n" ..
				"}\n", v[1], v[2], v[3])
	end
	return s
end

local function povprefix(camlocation, camat,lights)
	return string.format(
"background { color rgbf <1, 1, 1, 1> }\n" ..
"\n" ..
"camera {\n" ..
"   perspective\n" ..
"   location <%f, %f, %f>\n" ..
"   look_at  <%f, %f, %f>\n" ..
"}\n" ..
"\n" ..
_lights(lights) ..
"#declare arrow = difference {\n" ..
"  merge {\n" ..
"    cone {\n" ..
"      <0, 0.3, 0>, 0,\n" ..
"      <0, -0.5, 0>, 0.5\n" ..
"      scale 1\n" ..
"      rotate <0, 0, 0>\n" ..
"      translate y*1\n" ..
"      hollow false\n" ..
"    }\n" ..
"\n" ..
"    cylinder {\n" ..
"      <0, 0.5, 0>, <0, -0.5, 0>, 0.2\n" ..
"      scale 1\n" ..
"      rotate <0, 0, 0>\n" ..
"      translate <0, 0, 0>\n" ..
"      hollow false\n" ..
"    }\n" ..
"    translate <0, 0, 0>\n" ..
"    rotate <0, 0, 0>\n" ..
"    scale 1.5\n" ..
"  }\n" ..
"\n" ..
"  box {\n" ..
"    <-0.5, -0.5, -0.5>, <0.5, 0.5, 0.5>\n" ..
"    scale <2, 0.3, 2>\n" ..
"    rotate <0, 0, 0>\n" ..
"    translate y*1.3\n" ..
"  }\n" ..
"  translate <0,-0.5,0>\n" ..
"}\n" ..
"\n" ..
"#declare collar = intersection {\n" ..
"   merge {\n" ..
"         cone {\n" ..
"            <0, 0.3, 0>, 0,\n" ..
"            <0, -0.5, 0>, 0.5\n" ..
"            scale 1\n" ..
"            rotate <0, 0, 0>\n" ..
"            translate y*1\n" ..
"            hollow false\n" ..
"         }\n" ..
"\n" ..
"         cylinder {\n" ..
"            <0, 0.5, 0>, <0, -0.5, 0>, 0.2\n" ..
"            scale 1\n" ..
"            rotate <0, 0, 0>\n" ..
"            translate <0, 0, 0>\n" ..
"            hollow false\n" ..
"         }\n" ..
"         translate <0, 0, 0>\n" ..
"         rotate <0, 0, 0>\n" ..
"         scale 1.5\n" ..
"  }\n" ..
"\n" ..
"  box {\n" ..
"     <-0.5, -0.5, -0.5>, <0.5, 0.5, 0.5>\n" ..
"     scale <2, 0.3, 2>\n" ..
"     rotate <0, 0, 0>\n" ..
"     translate y*1.3\n" ..
"  }\n" ..
"\n" ..
"  texture {\n" ..
"     pigment {\n" ..
"        color rgb <0.980392, 0.980392, 0.980392>\n" ..
"     }\n" ..
"  }\n" ..
"  translate <0,-0.5,0>\n" ..
"}",
	camlocation[1], camlocation[2], camlocation[3],
	camat[1], camat[2], camat[3])
end


local function spin(mx, my, mz, x, y, z, r, g, b)

	local rad = (mx^2 + my^2 + mz^2)^(1/2)
	if rad == 0 then
		return ""
	end
	local theta = math.atan2(my, mx)
	local phi   = math.acos(mz/rad)

	local scale = rad*0.5

	return string.format(
		"merge {\n"                                    ..
		"   object{collar}\n"                          ..
		"\n"                                           ..
		"   object{arrow\n"                            ..
		"      texture {\n"                            ..
		"         pigment {\n"                         ..
		"            color rgb <%f,%f,%f>\n"           ..
		"         }\n"                                 ..
		"      }\n"                                    ..
		"   }\n"                                       ..
		""                                             ..
		"   scale %f\n"                               ..
		"   rotate<%f, 0, 0,>\n"                       ..
		"   rotate<0, %f, 0>\n"                        ..
		"   rotate y*+90\n"                            ..
		"   translate <%f, %f, %f>\n"                  ..
		"}\n", 
				r, g, b, 
				scale,
				phi * 180/math.pi, 
				theta * -180/math.pi,
				x,y,z)
end

local function range(t,i)
	local _min = t[1][i]
	local _max = t[1][i]

	for k,v in pairs(t) do
		if v[i] < _min then
			_min = v[i]
		end
		if v[i] > _max then
			_max = v[i]
		end
	end
	return _min, _max
end


function POVRay(filename, ss, cam_pos, cam_at, lights, color_func)
-- 	, {cam_atx, cam_aty, cam_atz}, 
--     {{light1x, light1y, light1z}, {light2x, light2y, light2z}, {light3...}, ...}},  
--      optional_colour_func )
--  function make_povray(filename, ss, six_col)
	color_func = color_func or colormap

	cam_pos = cam_pos or {20,20,20}
	cam_at = cam_at or {ss:nx()/2+1,ss:nz()/2+1,ss:ny()/2+1}
	lights = lights or {{20,20,0}, {0, 20,20}, {-20,20,0}, {0,20,-20}}
	
	local pov  = io.open(filename, "w")

	if pov == nil then
		error("Failed to open `" .. filename .. "' for writing")
	end
	
	local at = cam_at
	local lightpos = lights
	local cam = cam_pos

	pov:write( povprefix(cam_pos, cam_at, lights) )

	local c = 1
	for z=1,ss:nz() do
		for y=1,ss:ny() do
			for x=1,ss:nx() do
				local sx, sy, sz = ss:spin(x,y,z)

				local r, g, b = color_func(sx, sy, sz) 
				pov:write(spin(sx, sy, sz, x,z,y, r, g, b)) --coord twiddle
				c = c + 1
			end
		end
	end
	pov:close()
end




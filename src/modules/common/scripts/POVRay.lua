-- This file provides a function that converts a *SpinSystem* into a file that is ready to be rendered with the POV-Ray raytracer. The function is POVRay(filename, ss, custom) where filename is the output filename that will be written to, ss is the *SpinSystem* and custom is an optional table interpreted as a JSON style table. The accepted keys-values in the custom table are:
-- <pre>
-- camera_position = {camx, camy, camz}
-- camera_focus = {cam_atx, cam_aty, cam_atz}
-- lights = {{light1x, light1y, light1z}, {light2x, light2y, light2z}, {light3...}, ...}},  
-- color_function = function(sx,sy,sz) return r,g,b end
-- position_function = function(x,y,z) return x,y,z end
-- scale = 1.0
-- </pre>
-- 
-- Note: scale can be a number, an array or a function that takes site positions as input and returns the scale
-- 
-- Example Usage:
-- <pre>
-- dofile("maglua://POVRay.lua")
--
-- ss = SpinSystem.new(32,32)
-- ex = Exchange.new(ss)
-- llg = LLG.Cartesian.new(ss)
-- ss:setTimeStep(1e-3) 
--
-- function r()
--     return math.random()*2 - 1
-- end
--
-- for i=1,32 do
--     for j=1,32 do
--         ss:setSpin({i,j}, {r(), r(), r()}, 1)
--
--         ex:add({i,j}, {i+1,j}, 1)
--         ex:add({i,j}, {i-1,j}, 1)
--         ex:add({i,j}, {i,j+1}, 1)
--         ex:add({i,j}, {i,j-1}, 1)
--     end
-- end
--
-- function step()
--     ss:resetFields()
--     ex:apply(ss)
--     ss:sumFields()
--     llg:apply(ss)
-- end
--
-- while ss:time() < 3 do
--     step()
-- end
--
-- POVRay("example.pov", ss, {
--     camera_position = {16,15,40}, 
--     camera_focus = {16,16,0}, 
--     color_function = function(sx,sy,sz) 
--         local r = (sx+1)/2 
--         return r,r,r
--     end})


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
	local s = {}
	for k,v in pairs(tab) do
		table.insert(s, string.format(
				[[light_source {<%f,%f,%f>, rgb <1, 1, 1> fade_distance 100}]], v[1], v[2], v[3]))
	end
	return table.concat(s, "\n") .. "\n"
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

[[
#declare arrow = difference {
  merge {
    cone {
      <0, 0.3, 0>, 0,
      <0, -0.5, 0>, 0.5
      scale 1
      rotate <0, 0, 0>
      translate y*1
      hollow false
    }

    cylinder {
      <0, 0.5, 0>, <0, -0.5, 0>, 0.2
      scale 1
      rotate <0, 0, 0>
      translate <0, 0, 0>
      hollow false
    }
    translate <0, 0, 0>
    rotate <0, 0, 0>
    scale 1.5
  }

  box {
    <-0.5, -0.5, -0.5>, <0.5, 0.5, 0.5>
    scale <2, 0.3, 2>
    rotate <0, 0, 0>
    translate y*1.3
  }
  translate <0,-0.5,0>
}

#declare collar = intersection {
  merge {
    cone {
      <0, 0.3, 0>, 0,
      <0, -0.5, 0>, 0.5
      scale 1
      rotate <0, 0, 0>
      translate y*1
      hollow false
    }
    cylinder {
      <0, 0.5, 0>, <0, -0.5, 0>, 0.2
      scale 1
      rotate <0, 0, 0>
      translate <0, 0, 0>
      hollow false
    }
    translate <0, 0, 0>
    rotate <0, 0, 0>
    scale 1.5
  }

  box {
    <-0.5, -0.5, -0.5>, <0.5, 0.5, 0.5>
    scale <2, 0.3, 2>
    rotate <0, 0, 0>
    translate y*1.3
  }

  texture {
    pigment {
      color rgb <0.980392, 0.980392, 0.980392>
    }
  }
  translate <0,-0.5,0>
}]],
	camlocation[1], camlocation[2], camlocation[3],
	camat[1], camat[2], camat[3]) ..
	
	
[[	
#macro Make_Spin(RR,GG,BB,SS,RR1,RR2,XX,YY,ZZ)
merge {
  object{collar}
  object{arrow
    texture {
      pigment {
        color rgb <RR,GG,BB>
      }
    }
  }
  scale SS
  rotate<RR1, 0, 0,>
  rotate<0, RR2, 0>
  rotate y*+90
  translate <XX, YY, ZZ>
}
#end
]]
end


local function spin(mx, my, mz, x, y, z, r, g, b)

	local rad = (mx^2 + my^2 + mz^2)^(1/2)
	if rad == 0 then
		return ""
	end
	local theta = math.atan2(my, mx)
	local phi   = math.acos(mz/rad)

	local scale = rad

	return string.format("Make_Spin(% f,% f,% f, %g, %g, %g, %g,%g,%g)\n",
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


local function pos_func(x,y,z)
	return x,y,z
end

function POVRay(filename, ss, custom)
-- 	, {cam_atx, cam_aty, cam_atz}, 
--     {{light1x, light1y, light1z}, {light2x, light2y, light2z}, {light3...}, ...}},  
--      optional_colour_func )
--  function make_povray(filename, ss, six_col)
	custom = custom or {}
	local color_func = custom["color_function"] or colormap
	local pos = custom["position_function"] or pos_func

	local nx, ny, nz = ss:nx(), ss:ny(), ss:nz()
	local nn = (nx*nx+ny*ny+nz*nz)^(1/2)
	local cam_pos = custom.camera_position or {nx/2+1, -(1/8) * nn, (3/4) * nn}
	local cam_at = custom.camera_focus or {nx/2+1 ,ny/2+1 - (1/8)*nn, nz/2}
	local lights = custom.lights or {{20,0,20}, {0, 20,20}, {-20,0,20}, {0,-20,20}}
	local scale = custom.scale or 1
	
	local function switch_yz(t)
		return {t[1], t[3], t[2]}
	end

	cam_pos = switch_yz(cam_pos)
	cam_at  = switch_yz(cam_at)

	for k,v in pairs(lights) do
		lights[k] = switch_yz(lights[k])
	end

	local pov  = io.open(filename, "w")

	if pov == nil then
		error("Failed to open `" .. filename .. "' for writing")
	end
	
	local at = cam_at
	local lightpos = lights
	local cam = cam_pos

	pov:write( povprefix(cam_pos, cam_at, lights) )

	local pov_string = {}

	if type(scale) == "number" then -- not an array
		scale = scale * 0.5
		for z=1,nz do
			for y=1,ny do
				for x=1,nx do
					local sx, sy, sz, mm = ss:spin(x,y,z)

					if mm*scale > 1e-8 then
						local xx,yy,zz = pos(x,y,z)
						local r, g, b = color_func(sx, sy, sz) 
						table.insert(pov_string, spin(sx*scale, sy*scale, sz*scale, xx,zz,yy, r, g, b)) --coord twiddle
					end
				end
			end
		end
	else -- scale may be an array. we'll see!
		if type(scale) == "function" then
			local sf = scale --function
			for z=1,nz do
				for y=1,ny do
					for x=1,nx do
						local sx, sy, sz, mm = ss:spin(x,y,z)
						local scale = sf(x,y,z) * 0.5

						if mm*scale > 1e-8 then
							local xx,yy,zz = pos(x,y,z)
							local r, g, b = color_func(sx, sy, sz) 
							table.insert(pov_string, spin(sx*scale, sy*scale, sz*scale, xx,zz,yy, r, g, b)) --coord twiddle
						end
					end
				end
			end	
		else
			local sa = scale --array
			for z=1,nz do
				for y=1,ny do
					for x=1,nx do
						local sx, sy, sz, mm = ss:spin(x,y,z)
						local scale = sa:get(x,y,z) * 0.5

						if mm*scale > 1e-8 then
							local xx,yy,zz = pos(x,y,z)
							local r, g, b = color_func(sx, sy, sz) 
							table.insert(pov_string, spin(sx*scale, sy*scale, sz*scale, xx,zz,yy, r, g, b)) --coord twiddle
						end
					end
				end
			end
		end
	end
	pov:write( table.concat(pov_string, "\n"))

	pov:close()
end




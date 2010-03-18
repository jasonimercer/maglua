function _lights(tab)
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

function povprefix(camlocation, camat,lights)
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


function spin(mx, my, mz, x, y, z, r, g, b)

	local theta = math.atan2(my, mx)
	local phi   = math.acos(mz)

	scale = math.sqrt(mx^2 + my^2 + mz^2) * 0.4

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

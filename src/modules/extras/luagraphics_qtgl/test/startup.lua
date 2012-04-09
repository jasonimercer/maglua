--t = loadModule("../../debug/luagraphics_qtgl.dll")
t = loadModule("../libluagraphics_qtgl.so")
for k,v in pairs(t) do
	print(k,v)
end
scene = QGraphicsSceneLua.new(scene_userdata)

drawOpenGL = DrawOpenGL.new()

dofile("DrawSystem.lua")
dofile("Draw.lua")

slides = {}
widgets = {}

dofile("helper_function.lua")
dofile("slide01.lua")
dofile("slide02.lua")
dofile("slide03.lua")
dofile("slide04.lua")
dofile("slide05.lua")
dofile("slide06.lua")
dofile("slide07.lua")
dofile("slide08.lua")

drawfunc = {}
centerfunc = {}
current_slide = 1
slide_x = 2000
drawfunc[1], centerfunc[1] = makeSlide01(slide_x,0)slide_x=slide_x+2000
drawfunc[2], centerfunc[2] = makeSlide02(slide_x,0)slide_x=slide_x+2000
drawfunc[3], centerfunc[3], ss = makeSlide03(slide_x,0)slide_x=slide_x+2000
drawfunc[4], centerfunc[4] = makeSlide04(slide_x,0, ss)slide_x=slide_x+2000
drawfunc[5], centerfunc[5] = makeSlide05(slide_x,0, ss)slide_x=slide_x+2000
drawfunc[6], centerfunc[6] = makeSlide06(slide_x,0, ss)slide_x=slide_x+2000
drawfunc[7], centerfunc[7] = makeSlide07(slide_x,0, ss)slide_x=slide_x+2000
drawfunc[8], centerfunc[8] = makeSlide08(slide_x,0, ss)slide_x=slide_x+2000


function showSlide(i)
	if drawfunc[i] then
		setSceneDrawFunction(drawfunc[i])
	end
	if centerfunc[i] then
		centerfunc[i]()
	end
end

function next()
	current_slide = current_slide + 1
	if drawfunc[current_slide] == nil then
		current_slide = 1
	end
	showSlide(current_slide)
end

function prev()
	current_slide = current_slide - 1
	if drawfunc[current_slide] == nil then
		current_slide = table.maxn(drawfunc)
	end
	showSlide(current_slide)
end

showSlide(current_slide)



function key_func(abc)
end

setKeyFunction(key_func)

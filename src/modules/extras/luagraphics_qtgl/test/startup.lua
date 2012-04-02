-- t = loadModule("../../release/luagraphics_qtgl.dll")

t = loadModule("../libluagraphics_qtgl.so")

for k,v in pairs(t) do
	print(k,v)
end

dofile("DrawSystem.lua")
dofile("Draw.lua")

scene = QGraphicsSceneLua.new(scene_userdata)

te_x = QTextEditItemLua.new(scene)
te_x:setText("0")
te_x:resize(32, 24)
te_x:move(0,0)
te_x:setScrollBarPolicy(1)

te_y = QTextEditItemLua.new(scene)
te_y:setText("0")
te_y:resize(32, 24)
te_y:move(48,0)
te_y:setScrollBarPolicy(1)


function f()
	centerOn( te_x:text(), te_y:text() ) 
end
pb = QPushButtonItemLua.new(scene, "centerOn", f)
pb:move(96,0)
pb:setWidth(128)
pb:setRepeat(true)


function change_x(i)
	local field_x = (i-500)/100
	zee:set(field_x,0,0)
end

slider_x = QSliderItemLua.new(scene, change_x)
slider_x:setRange(0,1000)
slider_x:setValue(500)
slider_x:move(0,100)
slider_x:setTransparent(0.5)
x,y,w,h = slider_x:geometry()
slider_x:setHorizontal()
slider_x:setGeometry(x,y,h,w)



function change_ex(i)
	local ex_str = i/100
	ex:setScale(ex_str)
end
slider_ex = QSliderItemLua.new(scene, change_ex)
slider_ex:setRange(0,100)
slider_ex:setValue(0)
slider_ex:move(0,200)
slider_ex:setTransparent(0.5)
x,y,w,h = slider_ex:geometry()
slider_ex:setHorizontal()
slider_ex:setGeometry(x,y,h,w)
	
function update_model()
	step()
-- 	step()
-- 	step()
-- 	step()
-- 	step()

	for i=1,n do
		for j=1,n do
			local x, y, z = ss:spin({i,j})
			theta = math.atan2(y, x)
			phi = math.acos(z)
			
			ssg[i][j].set(theta,phi)
		end
	end

end


d = DrawOpenGL.new()

function draw_opengl()
	update_model()
	d:reset()
	d:draw(c, l, all_arrows)
end

setSceneDrawFunction(draw_opengl)
centerOn(50,50)



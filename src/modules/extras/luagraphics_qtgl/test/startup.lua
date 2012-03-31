t = loadModule("../libluagraphics_qtgl.so")
for k,v in pairs(t) do
	print(k,v)
end

dofile("/home/jmercer/programming/c/maglua/src/modules/extras/luagraphics/DrawSystem.lua")
dofile("Draw.lua")


scene = QGraphicsSceneLua.new(scene_userdata)

te_x = QTextEditItemLua.new(scene)
te_x:setText("0")
te_x:resize(32, 24)
te_x:move(0,0)
te_x:setScrollBarPolicy(1)

te_y = QTextEditItemLua.new(scene)
te_y:setText("0")
te_y:resize(32, 32)
te_y:move(48,0)
te_y:setScrollBarPolicy(1)


function f()
	centerOn( te_x:text(), te_y:text() ) 
end
pb = QPushButtonItemLua.new(scene, "centerOn", f)
pb:move(96,0)
pb:setWidth(128)
pb:setRepeat(true)



function update_model()
	step()

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

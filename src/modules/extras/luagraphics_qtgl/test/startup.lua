t = loadModule("../libluagraphics_qtgl.so")
for k,v in pairs(t) do
	print(k,v)
end

scene = QGraphicsSceneLua.new(scene_userdata)

te = QTextEditItemLua.new(scene)
te:setText("test")

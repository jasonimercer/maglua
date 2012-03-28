d = DrawPOVRay.new("file.pov")

s = Sphere.new()
s:setColor(Color.new(1,0,0))

ts = Scale.new(1, .01, 1)
ts:add(s)

l = Light.new()
l:setPosition(3,-5,3)

c = Camera.new()

d:draw(c, l, ts)
-- d:draw(l)
-- d:draw(c)


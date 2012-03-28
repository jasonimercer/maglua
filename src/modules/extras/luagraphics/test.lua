d = DrawPOVRay.new("file.pov")

function rand(a,b)
	b = b or 1
	a = a or 0
	return (b-a) * math.random() + a
end

function rcol()
	local red = r(0,1.5)
	local green = r(0,1.5-red)
	local blue = 1.5-red-green
	return red, green, blue
end

ts = Scale.new(1, 1, 1)
tr = Rotate.new(0,0,0)
tt = Translate.new(5,0,0)

-- local r, n = 2, 1024
-- -- for p=0,n/2 do
-- -- 	local phi = math.pi * 2*p/n
-- -- 	for t=1,n do
-- -- 		local theta = math.pi * 2 * t / n
-- for i=1,n do
-- 	theta = rand(0,2*math.pi)
-- 	phi = rand(0,math.pi)
-- 		local x,y,z
-- 		x = r * math.cos(theta) * math.sin(phi)
-- 		y = r * math.sin(theta) * math.sin(phi)
-- 		z = r * math.cos(phi)
-- 
-- 		tube = Tube.new()
-- -- 		s = Sphere.new()
-- 		local c = Color.new()
-- 		c:map(theta, phi)
-- 		tube:setColor(c)
-- 		tube:setPosition(1,x,y,z)
-- 		tube:setPosition(2,x*1.1,y*1.1,z*1.1)
-- 		tube:setRadius(1,0.1)
-- 		tube:setRadius(2,0)
-- 		
-- 		ts:add(tube)
-- -- 	end
-- end

function arrow(x,y)
	local global_translate = Translate.new()
	local local_phi = Rotate.new()
	local local_theta = Rotate.new()
	local head_translate = Translate.new()
	
	local head1 = Tube.new()
	local head2 = Tube.new()
	local head3 = Tube.new()
	local body = Tube.new()
	local color = Color.new(1,0,0,1)
	local white = Color.new(1,1,1,1)
	
	local r = 0.5
	local p1, p2, p3 = 0.2, 0.45, 0.8
	
	head1:setPosition(1, 0, 0, 0.0)
	head1:setPosition(2, 0, 0, p1)
	head1:setRadius(1, r)
	head1:setRadius(2, r * (1 - p1/p3))
	head1:setColor(color)
	
	head2:setPosition(1, 0, 0, p1)
	head2:setPosition(2, 0, 0, p2)
	head2:setRadius(1, r * (1 - p1/p3))
	head2:setRadius(2, r * (1 - p2/p3))
	head2:setColor(white)

	head3:setPosition(1, 0, 0, p2)
	head3:setPosition(2, 0, 0, p3)
	head3:setRadius(1, r * (1 - p2/p3))
	head3:setRadius(2, r * (1 - p3/p3))
	head3:setColor(color)
	
	body:setPosition(1, 0, 0, -1/2)
	body:setPosition(2, 0, 0,  1/2)
	body:setRadius(1, 0.25)
	body:setRadius(2, 0.25)
	body:setColor(color)
	
	head_translate:set(0,0,0.5)
	head_translate:add(head1,head2,head3)

	local_phi:add(head_translate, body)
	local_theta:add(local_phi)

	global_translate:add(local_theta)
	global_translate:set(x,y,0)
	
	return {global_translate, local_theta, local_phi, color}
end

n = 8
ss = SpinSystem.new(n,n)
ex = Exchange.new(ss)
zee = AppliedField.new(ss)
llg = LLG.Quaternion.new()
rng = Random.Isaac.new()
thermal = Thermal.new(ss)
thermal:set(0.5)

ss:setTimeStep(0.01)
zee:set(1,0,0)
for i=1,n do
	for j=1,n do
		ss:setSpin({i,j}, {rand(-1,1),rand(-1,1),rand(-1,1)}, 1)
		ex:add({i,j}, {i+1,j},1)
		ex:add({i,j}, {i-1,j},1)
		ex:add({i,j}, {i,j+1},1)
		ex:add({i,j}, {i,j-1},1)
	end
end

function step()
	ss:resetFields()
	zee:apply(ss)
	ex:apply(ss)
	thermal:apply(rng, ss)
	ss:sumFields()
	llg:apply(ss)
end

for i=1,1000 do
	step()
end

ssg = {}
local all_arrows = Translate.new()
for i=1,n do
	ssg[i] = {}
	for j=1,n do
		local x, y, z = ss:spin({i,j})
		theta = math.atan2(y, x)
		phi = math.acos(z)
		
		ssg[i][j] = arrow(i*2, j*2)
		
		ssg[i][j][3]:set(0,phi * 180/math.pi,0)
		ssg[i][j][2]:set(0,0,theta * 180/math.pi)
		ssg[i][j][4]:map(theta,phi)
		
		
		all_arrows:add(ssg[i][j][1])
	end
end


l = Light.new()
l:setPosition(5,-20,4)

c = Camera.new()
-- c:zoom(-10)
c:translateUVW(n,5,0)
c:rotate(0,-math.pi/4,0)

d:draw(c, l, all_arrows)
d:close()

os.execute("povray -W800 -H600 +A0.1 +P -UV file.pov")

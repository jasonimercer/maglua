n = 16
ss = SpinSystem.new(n,n)
ex = Exchange.new(ss)
dip = Dipole.new(ss)
zee = AppliedField.new(ss)
llg = LLG.Quaternion.new()
rng = Random.Isaac.new()
thermal = Thermal.new(ss)

thermal:set(0.01)
dip:setTruncation(1000)
dip:setStrength(1)

ss:setTimeStep(0.05)
zee:set(0,0,0)
ex:setScale(0)
for i=1,n do
	for j=1,n do
		ss:setSpin({i,j}, {1,0,0}, 1)
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
	dip:apply(ss)
	thermal:apply(rng, ss)
	ss:sumFields()
	llg:apply(ss)
end

-- for i=1,100 do
-- 	step()
-- end

all_arrows, ssg = makeSysetm(ss)

for i=1,n do
	for j=1,n do
		local x, y, z = ss:spin({i,j})
		theta = math.atan2(y, x)
		phi = math.acos(z)
		
		ssg[i][j].set(theta,phi)
	end
end


l = Light.new()
l:setPosition(5,-20,4)

c = Camera.new()
-- c:zoom(-10)
c:translateUVW(n/2,5,0)
c:rotate(0,-math.pi/4,0)

server = Server.new(55000)
server:share("ss", "ex", "zee", "dip", "thermal")
server:startBackground()

-- os.execute("povray -W800 -H600 +A0.1 +P -UV file.pov")

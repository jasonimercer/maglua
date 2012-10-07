dofile("maglua://RungeKutta.lua")
dofile("maglua://POVRay.lua")

ss  = SpinSystem.new(16,16)
ex  = Exchange.new(ss)
llg = LLG.Quaternion.new()

ss:setTimeStep(1e-2)

function r3()
	local function r()
		return math.random()*2-1
	end
	return {r(), r(), r()}
end

for i=1,16 do
	for j=1,16 do
		if i%4 > 0 then
			ss:setSpin({i,j}, r3(), 1)
		end
		
		ex:add({i,j}, {i+1,j}, 1)
		ex:add({i,j}, {i-1,j}, 1)
		ex:add({i,j}, {i,j+1}, 1)
		ex:add({i,j}, {i,j-1}, 1)
	end
end

function calcField(ss)
	ss:resetFields()
	ex:apply(ss)
	ss:sumFields()
end

step = make_rk_step_function(ss, "RK4", calcField, llg)

while ss:time() < 2 do
	step()
end


POVRay("SpinSystem.pov", ss)


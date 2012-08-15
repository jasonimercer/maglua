n = 16
ss = SpinSystem.new(n,n,n)
ex = Exchange.new(ss)
lr = LongRange3D.new(ss)
llg = LLG.Cartesian.new(ss)

J = 0.75
ss:setTimeStep(1e-3)

lr:loadTensors("16x16x16_cubic.lua")

math.randomseed(1000)
function r3()
	local r = function()
		return math.random()*2-1
	end
	return {r(), r(), r()}
end
for z=1,n do
	for y=1,n do
		for x=1,n do
			ex:add({x,y,z}, {x+1,y,z}, J)
			ex:add({x,y,z}, {x-1,y,z}, J)
			ex:add({x,y,z}, {x,y+1,z}, J)
			ex:add({x,y,z}, {x,y-1,z}, J)
			ex:add({x,y,z}, {x,y,z+1}, J)
			ex:add({x,y,z}, {x,y,z-1}, J)
			
			if x >= n-4 or y >= n-4 or z >= n-4 then
				ss:setSpin({x,y,z}, r3(), 1)
			else
				ss:setSpin({x,y,z}, r3(), 0)
			end
		end
	end
end


for x=5,8 do
	for y=5,8 do
		for z=1,n do
			ss:setSpin({x,y,z}, r3(), 1)
			ss:setSpin({y,z,x}, r3(), 1)
			ss:setSpin({z,x,y}, r3(), 1)
		end
	end
end

function step()
	ss:resetFields()
	ex:apply(ss)
	lr:apply(ss)
	ss:sumFields()
	
	llg:apply(ss)
end

s = 1 
function save()
	local filename = string.format("checkpoint_%04d.dat", s)
	s = s + 1
	print(ss:time(), filename)
	checkpointSave(filename, ss)
end

next_report = 0
while ss:time() < 10 do
	step()

	if ss:time() >= next_report then
		save()
		next_report = next_report + 0.5
	end
end
	
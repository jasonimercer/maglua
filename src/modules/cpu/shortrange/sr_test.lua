dofile("maglua://CGS.lua")

ss  = SpinSystem.new(16,16,2)
mag = Magnetostatics2D.new(ss)


A, B, C = {10*nm, 0, 0}, {0, 10*nm, 0}, {0*nm,0*nm,10*nm}

mag:setUnitCell(1, A, B, C)
mag:setUnitCell(2, A, B, C) --this one doesn't matter

mag:setGrainSize(1, 10*nm, 10*nm, 10*nm)
mag:setGrainSize(2, 10*nm, 10*nm, 10*nm) -- this does matter

mag:setTruncation(2,2,2,2) -- very small truncation. We want to try to replicate it with a Sort Range operator

cell1 = 10*10*10*nm^3
cell2 = 10*10*10*nm^3

for x=1,16 do
    for y=1,16 do
	ss:setSpin({x,y,1}, {0,0,1}, (500*emu/cc)* cell1)
	ss:setSpin({x,y,2}, {0,0,1}, (500*emu/cc)* cell2)
    end
end
ss:setSpin({9,8,1}, {.5,.1, 0}, (500*emu/cc)* cell1) -- adding in some texture
ss:setSpin({9,8,2}, {.7,.3,.8}, (500*emu/cc)* cell2)


mag:setStrength(1,(4*math.pi)/(cell1)) -- scale factor for CGS
mag:setStrength(2,(4*math.pi)/(cell2))

mag:apply(ss)





sr = ShortRange.new(ss)


-- reproducing entire mag2d at 8,8,1
for x=8-2,8+2 do
    for y=8-2,8+2 do
	for z=1,2 do
	    sr:addMagnetostatic2D(mag, {x,y,z}, {8,8,1}, "Calculate")
	    sr:addMagnetostatic2D(mag, {x,y,z}, {8,8,2}, "Calculate")
	end
    end
end
sr:apply(ss)

function printFieldAt(pos, name)
    local hx = ss:fieldArrayX(name):get(pos)
    local hy = ss:fieldArrayY(name):get(pos)
    local hz = ss:fieldArrayZ(name):get(pos)
    print(string.format("%30s {%2i,%2i,%2i} = {% 7g, % 7g, % 7g}", tostring(name),pos[1],pos[2],pos[3],hx,hy,hz))
end


for z=1,2 do
    print()
    printFieldAt({8,8,z}, mag)
    printFieldAt({8,8,z}, sr)
end

--[[
local v1 = sr:addMagnetostatic2D(mag, {9,10,2}, {8,8,1}, "Calculate")
local v2 = sr:addMagnetostatic2D(mag, {9,10,2}, {8,8,1}, "Read")
for k,v in pairs(v1) do
    print(string.format("%s   % 11g    % 11g", k,v1[k],v2[k]))
end
--]]

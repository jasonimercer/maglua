-- 
-- This test script divides a slab of Magnetostatically coupled
-- blocks into various Z levels and computes the resulting energy.
-- If the energy is not constant then there is a problem with the
-- underlying code.
-- 
-- Thre are 4 test cases
--   1: No splits
--   2: Split at the mid point
--   3: Split at the 1/3 point
--   4: 2 Splits at 1/4 and 5/8

zSplits = {{}, {1/2}, {1/3}, {1/4, 5/8}}
for i=1,4 do --tack on a 0 and a 1 for clean expressions later
	table.insert(zSplits[i], 1)
	zSplits[i][0] = 0
end

dofile("maglua://CGS.lua")
Ms = 1400 * emu/cc

a, b, c = 4*nm, 5*nm, 8*nm

-- non-trivial initial state
function momentDirection(x,y)
  if x == 1 and y == 1 then
    return {0,0,1}
  end
  local theta = (x/4 + y/5) * 2 * math.pi
  return {math.cos(theta), math.sin(theta), 0}
end

function getCell(test_number, layer)
  local z  = zSplits[test_number]
  return a, b, c * (z[layer] - z[layer-1])
end

print("Test\tEnergy (erg)") --results header
print("---------------------------")
for test = 1,4 do
  layers = table.maxn(zSplits[test])

  ss = SpinSystem.new(4,4,layers)
  mag = Magnetostatics2D.new(ss)

  for x=1,4 do
    for y=1,4 do
      for z=1,layers do
		local cx,cy,cz = getCell(test, z)
        ss:setSpin({x,y,z}, momentDirection(x,y), Ms*cx*cy*cz)
	  end
	end
  end

  for z=1,layers do
    local cx,cy,cz = getCell(test, z)
    mag:setUnitCell(z, {cx,0,0},{0,cy,0},{0,0,cz})
	mag:setGrainSize(z, cx,cy,cz)
	mag:setStrength(z, (4*math.pi)/(cx*cy*cz))
  end
  mag:setTruncation(25) --this is low but will work for this case
  
  mag:apply(ss)
  
  local ex = ss:fieldArrayX("LongRange"):dot(ss:spinArrayX())
  local ey = ss:fieldArrayY("LongRange"):dot(ss:spinArrayY())
  local ez = ss:fieldArrayZ("LongRange"):dot(ss:spinArrayZ())
  
  print(test, ex+ey+ez)
end

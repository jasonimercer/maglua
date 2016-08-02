small = {1,1/2,1/4}
  big = {2,1,1/4}
  
function test(f)
	v1 = f(-2,-2,-1, small, small)
	   + f(-1,-2,-1, small, small)
	   + f(-2,-1.5,-1, small, small)
	   + f(-1,-1.5,-1, small, small)
	
	v2 = f(-2,-2,-1, small, big)

	return v1, v2
end

function same(a,b)
	local x = math.abs(a-b) / math.abs(a)
	if x < 0.00001 then
		return "Y  1.000\t"
	end
	return string.format("N % 4.3f\t", a/b)
end

print("NXX", "\t\tNXY", "\t\tNXZ")
a1,b1 = test(Magnetostatics2D.NXX)
a2,b2 = test(Magnetostatics2D.NXY)
a3,b3 = test(Magnetostatics2D.NXZ)
print(a1,a2,a3)
print(b1,b2,b3)
print(same(a1,b1), same(a2,b2), same(a3,b3))
print()

print("NYX", "\t\tNYY", "\t\tNYZ")
a1,b1 = test(Magnetostatics2D.NYX)
a2,b2 = test(Magnetostatics2D.NYY)
a3,b3 = test(Magnetostatics2D.NYZ)
print(a1,a2,a3)
print(b1,b2,b3)
print(same(a1,b1), same(a2,b2), same(a3,b3))
print()

print("NZX", "\t\tNZY", "\t\tNZZ")
a1,b1 = test(Magnetostatics2D.NZX)
a2,b2 = test(Magnetostatics2D.NZY)
a3,b3 = test(Magnetostatics2D.NZZ)
print(a1,a2,a3)
print(b1,b2,b3)
print(same(a1,b1), same(a2,b2), same(a3,b3))
print()
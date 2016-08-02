-- print(os.hostname(), os.domainname())
-- print(os.pid())
-- 
-- for k,v in pairs(os.uname()) do
-- 	print(k,v)
-- end


-- t1 = {1,2,3, 8}
-- t2 = {2,3,4}

-- print(math.dot(t1,t2))

rng = Random.Isaac.new()

-- a = {1,0,0}
-- b = {0,0,1}
a = {rng:normal(), rng:normal(), rng:normal()}

b = {rng:normal(), rng:normal(), rng:normal()}

a = math.scaledVector(a, 1/math.norm(a))
b = math.scaledVector(b, 1/math.norm(b))


n = math.cross(a,b) 

t = math.angleBetween(a, b)

c = math.rotateAboutBy(a, n, -t)

print(t)

-- print(type(a))

-- c = math.cross(a, b)

-- c = math.rotateAboutBy(a, b, math.pi/4)

for k,v in pairs(b) do
	print(k,v)
end
print()
for k,v in pairs(c) do
	print(k,v)
end

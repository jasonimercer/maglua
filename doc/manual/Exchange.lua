-- Section: Exchange
--
-- An Exchange operator keeps track of exchange interactions between spins and 
-- calculates the resulting fields. Lets create a new SpinSystem and 
-- corresponding Exchange Operator.

nx, ny = 8, 8
ss = SpinSystem.new(nx, ny)
ex = Exchange.new(nx, ny)

-- We will orient all the spins along the +x axis


for j=1,ny do
	for i=1,nx do
		ss:setSpin({i,j}, {1,0,0})
	end
end

-- Now we can define exchange pathways from one spin to another. This is done 
-- using the "addPath" method which takes two lattice sites and 1 interaction 
-- strength. Lets make all spins interact with the spin on their "right" with a 
-- strength of 1.

for j=1,ny do
	for i=1,nx do
		ex:addPath({i,j}, {i+1, j}, 1)
	end
end

-- Since we told the Exchange object the size of the lattice, it has 
-- automatically converted out of bound lattice sites to valid sites by applying
-- periodic boundary conditions. 
-- 
-- Lets complete our Exchange operator by adding in the other 3 nearest 
-- neighbour pathways

for j=1,ny do
	for i=1,nx do
		ex:addPath({i,j}, {i-1, j}, 1)
		ex:addPath({i,j}, {i, j+1}, 1)
		ex:addPath({i,j}, {i, j-1}, 1)
	end
end

-- You should not that a pathway is a directed connection, that is, adding a 
-- path from "a" to "b" does not imply that the path from "b" to "a" has been 
-- added. 
-- 
-- Our exchange operator is now initialized. It can compute the fields for 
-- nearest neighbour, in-plane, ferromagnetic interactions for an 8x8 lattice. 
-- Lets run that calculation by applying the Exchange object to the SpinSystem.

ex:apply(ss)

-- We can query the fields due to interactions with the SpinSystem "getField" 
-- method. Lets find the Exchange field at location (1,1)

fx, fy, fz = ss:getField("Exchange", {1,1})
print("Exchange Field at (1,1):", fx, fy, fz)

-- which seems to be reasonable.

--
-- This script tests tensor elements. As a base we will construct 
-- larger volumes with groups of same size blocks. Then we will 
-- look at reproducing the same elements with larger blocks 
-- interacting with the original small block. If the tensor
-- generation code is right then the results should be the same.
--
-- Creating a table of tensor functions for convenience

t = {
  XX=Magnetostatics2D.NXX,
  XY=Magnetostatics2D.NXY,
  XZ=Magnetostatics2D.NXZ,
  YX=Magnetostatics2D.NYX,
  YY=Magnetostatics2D.NYY,
  YZ=Magnetostatics2D.NYZ,
  ZX=Magnetostatics2D.NZX,
  ZY=Magnetostatics2D.NZY,
  ZZ=Magnetostatics2D.NZZ
}

-- The tensor functions take an offset of 3 numbers
-- and 2 tables. The first table is the x,y,z dimensions
-- of "destination" block, the second is the x,y,z 
-- dimensions of the "source" block. The offsets are 
-- defined from each block's origin which is one of the
-- block corners. Remember that offsets are not defined 
-- from the block centers. 
--
-- First we will define a base block size, this will be
-- used throughout the tests. We will pick dimensions so
-- we wont trigger fortuitious alignments of one 
-- dimension being a linear combination of others. In
-- real applications built with these tensors they can be
-- multiples of each other or straight cubes. We want a 
-- thorough test here so we're picking values that will
-- expose any bugs. 

block = {1,2^(1/2),2^(1/3)}

-- This is the function that will run the tests
function test(dest_block, source_group, source_single)
	local group_result = {}
	local single_result = {}

	-- iterating over 9 tensor functions
	for ab,func in pairs(t) do
		single_result[ab] = 0
		group_result[ab] = 0

		-- iterating over group of small source blocks
		for i=1,table.maxn(source_group) do
			local r = source_group[i][1]
			local b = source_group[i][2]
			group_result[ab] = group_result[ab] 
				+ func(r[1], r[2], r[3], dest_block, b)
		end

		local r = source_single[1]
		local b = source_single[2]
		single_result[ab] = func(r[1], r[2], r[3], dest_block, b)
	end
	
	local x = {
		{"XX", "XY", "XZ"},
		{"YX", "YY", "YZ"},
		{"ZX", "ZY", "ZZ"}
	}
	
	-- printing results in a formated table
	print("\tDisplacement = " .. table.concat(source_single[1], ", "))
	for i=1,3 do
		local s,g,d = {},{},{}
		for j=1,3 do
			s[j] = single_result[x[i][j]]
			g[j] =  group_result[x[i][j]]
			d[j] = s[j] - g[j]

			s[j] = string.format("% e", s[j])
			g[j] = string.format("% e", g[j])
			d[j] = string.format("% e", d[j])
		end
		print("    Component", table.concat(x[i], "\t\t"))
		print(" Sum of Group", table.concat(g, "\t"))
		print("Large Element", table.concat(s, "\t"))
		print("   Difference", table.concat(d, "\t"))
	end
end 


print("\nThe following tables show tensor elements of magnetostatic interactions")
print("between different sized volumes. One way is done by summing together a group")
print("of similar sized volumes, the other is calculated using different sized blocks.")

print("\nIn the following tests, the difference rows should be at least 12 orders")
print("of magnitude smaller than the calculated tensor elements otherwise there is")
print("a problem with the code.")

print("\n\nTest 1: Three blocks side by side along the X direction.")
rx,ry,rz = 5, 4, 0-1
group = {
	{{rx,            ry, rz}, block},
	{{rx+1*block[1], ry, rz}, block},
	{{rx+2*block[1], ry, rz}, block}}
large = {{rx,ry,rz}, {block[1]*3, block[2], block[3]}}

test(block, group, large) 

print("\n\nTest 2: Four blocks in a 2x2 grid in the XY plane.")
rx,ry,rz = -10, 3.5, 0.1
group = {
	{{rx,          ry,          rz}, block},
	{{rx,          ry+block[2], rz}, block},
	{{rx+block[1], ry+block[2], rz}, block},
	{{rx+block[1], ry,          rz}, block}}
large = {{rx,ry,rz}, {block[1]*2, block[2]*2, block[3]}}
test(block, group, large) 


print("\n\nTest 3: Twelve blocks in a 2x2x3 grid.")
rx,ry,rz = -1, 1.5, -3.15
group = {
	{{rx,          ry,          rz+0*block[3]}, block},
	{{rx,          ry+block[2], rz+0*block[3]}, block},
	{{rx+block[1], ry+block[2], rz+0*block[3]}, block},
	{{rx+block[1], ry,          rz+0*block[3]}, block},
	{{rx,          ry,          rz+1*block[3]}, block},
	{{rx,          ry+block[2], rz+1*block[3]}, block},
	{{rx+block[1], ry+block[2], rz+1*block[3]}, block},
	{{rx+block[1], ry,          rz+1*block[3]}, block},
	{{rx,          ry,          rz+2*block[3]}, block},
	{{rx,          ry+block[2], rz+2*block[3]}, block},
	{{rx+block[1], ry+block[2], rz+2*block[3]}, block},
	{{rx+block[1], ry,          rz+2*block[3]}, block}}

large = {{rx,ry,rz}, {block[1]*2, block[2]*2, block[3]*3}}
test(block, group, large) 





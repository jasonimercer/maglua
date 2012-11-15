-- this example script shows that you can checkpoint any type of data (except coroutines)

s = checkpointToString("a", function() print("hello") end, math.pi)

print(s)

x,y,z = checkpointFromString(s)

print(x,y,z)
y()

-- output:
-- M0TA%0TM03TE.5```````````````````````````````````````````````
-- M````````````````````````````````````````````````````````````
-- M```````````````````````````````````````````````````#````"@``
-- M``0````"````80"(````!@```(`````;3'5A40`!!`@$"``*`````````$!T
-- M97-T+FQU80`#`````P````````($````!0```$%````<0``!'@"```(````$
-- M!@````````!P<FEN=``$!@````````!H96QL;P``````!`````,````#````
-- @`P````,```````````````P````#````&"U$5/LA"4``
-- `
-- end
-- 
-- a	function: 0x1050a40	3.1415926535898
-- hello


-- print(x)

-- checkpointSave("cp.dat", 5)

-- y = checkpointLoad("cp.dat")
-- print(y)


if false then
function f(g,a,b)
	return g(a,b)
end

function add(a,b)
	return a+b
end
function sub(a,b)
	return a-b
end

print("Original Values:")
print(f(add,1,2))
print(f(sub,1,2))

print("Saving data and functions")
t = {1,2}
checkpointSave("test.dat", f, add, sub, t)
f, add, sub, t = nil, nil, nil, nil -- clearing old data

print("Loading data and functions")
g, plus, minus, p = checkpointLoad("test.dat")

print("Values from Load:")
print(g(plus, p[1], p[2]))
print(g(minus, p[1], p[2]))
end

long_variable_1 = 5
long_variable_2 = 6
long_variable_3 = 7
long_variable_4 = 8
long_variable_5 = 9

function gi(x)
    for k,v in pairs(debug.getinfo(x)) do
	print(k,v)
    end
end


local a = 4

function g()
    local q = 6
    local function f(x,y,z)
	local a = 5
	interactive("interactive")
	print(q,q,q)
	print(a,x,y,z)
    end
    return f
end


f = g()

f(1,2,3)
print("hello")
print(a)

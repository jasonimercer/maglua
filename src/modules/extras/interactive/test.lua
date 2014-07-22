
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

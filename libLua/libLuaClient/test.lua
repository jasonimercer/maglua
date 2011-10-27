function f(a, b)
	return func(), a+b
end

-- f(2, 3)

c = Client.new()
c:connect("localhost:55000")

print(c:remote(f, 2, 3))

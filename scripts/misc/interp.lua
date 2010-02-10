d = Interpolate.new()

r = 10
for i=0,r do
	t = i * math.pi * 2 / r
	d:addData(t, math.sin(t))
end

r = 100
for i=0,r do
	t = i * math.pi * 2 / r
	print(t, d:value(t))
end

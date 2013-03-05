print(os.hostname(), os.domainname())
print(os.pid())


print("CPU Affinity", table.concat(os.getCPUAffinity(), ", "))

	
for k,v in pairs(os.uname()) do
	print(k,v)
end

print(os.statm())

t = {}
for i=1,100000 do
	t[i] = i
end

print(os.statm())

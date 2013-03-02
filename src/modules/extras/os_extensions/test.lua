print(os.hostname(), os.domainname())
print(os.pid())


print("CPU Affinity", table.concat(os.getCPUAffinity(), ", "))

	
for k,v in pairs(os.uname()) do
	print(k,v)
end

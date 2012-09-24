print(os.hostname(), os.domainname())
print(os.pid())

for k,v in pairs(os.uname()) do
	print(k,v)
end
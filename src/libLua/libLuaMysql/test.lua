sql = MySQL.open("localhost", "jmercer", "thepasswd", "Test")

sql:exec("DROP TABLE IF EXISTS People;")
sql:exec("CREATE TABLE People(id INT AUTO_INCREMENT, name TEXT, age INT, PRIMARY KEY(id));")


sql:exec("INSERT INTO People(name,age) VALUES('Jason Mercer', 30);")
sql:exec("INSERT INTO People(name,age) VALUES('Louise Dawe', 31);")

a = sql:exec("SELECT name,age FROM People;")

for k,v in pairs(a) do
	for x,y in pairs(v) do
		print(k,x,y)
	end
end

-- sql:close()

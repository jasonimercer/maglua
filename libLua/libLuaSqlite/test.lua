sql = SQL.open("test.sqlite")

sql:exec("CREATE TABLE People(id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")


sql:exec("INSERT INTO People VALUES(NULL, 'Jason Mercer', 30);")
sql:exec("INSERT INTO People VALUES(NULL, 'Louise Dawe', 31);")

a = sql:exec("SELECT (name) FROM People;")

for k,v in pairs(a) do
	for x,y in pairs(v) do
		print(k,x,y)
	end
end

sql:close()
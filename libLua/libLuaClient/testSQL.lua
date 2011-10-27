function f(filename, b)
	local res
	local sql = SQL.open(filename)

	res = sql:exec("SELECT * FROM sqlite_master WHERE type='table' AND name='People';")

	if res[1] == nil then --doesn't exist, create it
		sql:exec("CREATE TABLE People(id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")

		sql:exec("INSERT INTO People VALUES(NULL, 'Jason Mercer', 30);")
		sql:exec("INSERT INTO People VALUES(NULL, 'Louise Dawe', 31);")
	end

	res = sql:exec("SELECT (name) FROM People;")

	sql:close()

	return res
end

c = Client.new("localhost:55000")

res = c:remote(f, "test2.sqlite")

for k,v in pairs(res) do
	for x,y in pairs(v) do
		print(k,x,y)
	end
end


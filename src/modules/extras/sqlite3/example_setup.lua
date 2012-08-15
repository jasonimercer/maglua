sql = SQLite3.new("example.sqlite")

sql:exec("CREATE TABLE IF NOT EXISTS People(id INTEGER PRIMARY KEY, name TEXT, age INTEGER, color TEXT, registered INTEGER);")

sql:exec("INSERT INTO People VALUES(NULL, 'Alice', 5, 'red',   0);")
sql:exec("INSERT INTO People VALUES(NULL, 'Bob',   6, 'orange',0);")
sql:exec("INSERT INTO People VALUES(NULL, 'Carla', 8, 'yellow',0);")
sql:exec("INSERT INTO People VALUES(NULL, 'Daniel',9, 'green', 0);")
sql:exec("INSERT INTO People VALUES(NULL, 'Ellen', 4, 'blue',  0);")
sql:exec("INSERT INTO People VALUES(NULL, 'Frank', 6, 'indigo',0);")
sql:exec("INSERT INTO People VALUES(NULL, 'Gloria',7, 'violet',0);")


-- bootstrap function, must be named "init" expected to take an active database
function init(s)
	local function checkout(sql) --custom checkout function for this data/problem
		local res = sql:exec("SELECT * FROM People WHERE registered=0 LIMIT 1;")

		if res[1] then
			sql:exec("UPDATE People SET registered=-1 WHERE registered=0 AND id="..res[1].id)
			if sql:changes() > 0 then
				return res[1].id
			end
		end
	end
	
	local function checkin(sql, id) --custom checkin function for this data/problem
		if id then
			sql:exec("UPDATE People SET registered=1 WHERE registered=-1 AND id="..id) --reg = -1 : pending
			return sql:changes() > 0
		end
		return false
	end
	
	return checkout, checkin
end

sql:setupBootstrap(init)

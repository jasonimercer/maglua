sql = SQLite3.new("example.sqlite")
checkout, checkin = sql:bootstrap()

id = checkout(sql)

if id then
    res = sql:exec("SELECT * FROM People WHERE id="..id..";")
    print(res[1].id, res[1].name)
end

checkin(sql, id)


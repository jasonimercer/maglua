--
-- these are additional methods implemented in lua for the SQLite3 module
-- 

local MODNAME = "SQLite3"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local function slines(txt) --inefficient
	local function _s(txt, t)
		local a, b, c, d = string.find(txt, "(.-)\n(.*)")
		if a then
			table.insert(t, c)
			return _s(d, t)
		end
		table.insert(t, txt)
		return t
	end
	return _s(txt, {})	
end

local function get_src(f)
	local t = debug.getinfo(f)
	if t["func"] == nil then
		return nil
	end

	if t["source"] == "=[C]" then
		return tostring(f)
	end
	
	local a, b = t["linedefined"], t["lastlinedefined"]
	local src = {}
	
	if string.sub(t["source"], 1, 1) == "@" then
		local g = io.open(string.sub(t["source"], 2))

		for i=1,a-1 do
			g:read("*line")
		end
		for i=a,b do
			table.insert(src, g:read("*line"))
		end
		g:close()
		src = table.concat(src, "\n")
		return src
	else
		t.source = string.gsub(t.source, "\r\n", "\n")
		local lines = slines(t.source)
		
		for i=a,b do
			table.insert(src, lines[i])
		end
		src = table.concat(src, "\n")
		return src
	end

	return nil
end



function bootstrap(sql)
	local res = sql:exec("SELECT code FROM Bootstrap;")
	local code = res[table.maxn(res)]["code"]
	local g = loadstring(code)
	g() --load init into scope
	return init(sql)
end

local function setupBootstrap(sql, f)
	sql:exec([[
		CREATE TABLE IF NOT EXISTS Bootstrap(version INTEGER PRIMARY KEY, code TEXT);
	]])
	
	local src = get_src(f)
	ee = "INSERT INTO Bootstrap VALUES(NULL,  '" .. sql:escapeString(src) .. "' );"
	sql:exec(ee)
	
end

t.bootstrap = bootstrap
t.setupBootstrap = setupBootstrap

local help = MODTAB.help

MODTAB.help = function(x)
	if x == bootstrap then
		return
		"Convenience method to execute the default bootstrap action on a database. The following function is interpreted as a method:\n<pre>" .. get_src(bootstrap) .. "</pre>\n",
		"",
		"The return values from the init function."
	end
	
	if x == setupBootstrap then
		return
		"Convenience method setup the default bootstrap action on a database",
		"1 Function: The function's code is added to the database the function name is expected " ..
		"to be \"init\" and will be passed the database on function call. The following function is interpreted as a method:\n<pre>" .. get_src(setupBootstrap) .. "</pre>\n",
		""
	end
	
	if x == nil then
		return help()
	end
	return help(x)
end


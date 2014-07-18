
local prompt = "> "

-- build an environment that allows the reading of locals and upvalues
-- would be nice to be able to write to them. not a current priority as
-- this is used mainly for debugging
function add_local_env(cmd, level)
    local locals = {}
    local env = debug.getinfo(level)

    while env do
 	local i,k,v = 1,debug.getlocal(level, 1)

	while k do
	    if locals[k] == nil then
		locals[k] = v
	    end
	    -- print("local:",i,k,v)
	    i,k,v = i+1,debug.getlocal(level, i+1)
	end

	if env.func then
	    i,k,v = 1,debug.getupvalue(env.func, 1)
	    while k do
		if locals[k] == nil then
		    locals[k] = v
		end
		-- print("upval:",i,k,v)
		i,k,v = i+1,debug.getupvalue(env.func, i)
	    end
	end

	level = level + 1
	env = debug.getinfo(level)
    end


    local f = loadstring(cmd)
    if f then
        local caller = debug.getinfo(f).func
        local fenv = debug.getfenv(caller)
        setmetatable(locals, { __index = fenv })
        setfenv(f, locals)
        return f
    end
end


local function get_input()
    return io.read()
end

local function loadline()
    local line = {}
    local chunk,err
    local keep_going = true

    while keep_going do
	io.write(prompt)
	local x = io.read()
	if x == nil then
	    return false -- done
	end

	table.insert(line, x)
	
	chunk, err = loadstring(table.concat(line, "\n"), "=stdin")

	if err then
	    keep_going = string.find(err, "'end' expected") ~= nil
	else
	    return add_local_env(table.concat(line, "\n"), 4)
	end
    end

    print(err)
    return function() end
end

local function bt(level) -- backtrace
    level = level or 1
    local trace = {}
    local env = debug.getinfo(level)

    while env do
	table.insert(trace, (env.short_src or "") .. ":" .. (env.currentline or 0))
	level = level + 1
	env = debug.getinfo(level)
    end


    -- peel off last 4 trace levels as it's MagLua botstrap and startup
    for i=1,4 do
	table.remove(trace)
    end

    for k,v in pairs(trace) do
	print(table.maxn(trace) - k + 1,v)
    end
    
end

function interactive(...)
    local caller = debug.getinfo(2)
    local src = caller.short_src or ""
    local line = caller.currentline or "0"


    print("Interactive environemnt initiated from (" .. src .. ":" .. line .. ")")
    print("Call Stack:")
    bt(3)
    print("Continue script with Ctrl+d")
    print(...)


    while true do
	local interactive_statement = loadline()
	if interactive_statement == false then
	    return
	end

	local stat, err = pcall(interactive_statement)

	if err then
	    print(err)
	end
    end

end

-- In this file we will add command line help switches and code

help_args = help_args or {}

table.insert(help_args, {"",""})
table.insert(help_args, {"Profile Related:",""})
table.insert(help_args, {"--profile name", "Profile the code. Results written to name_pid.txt"})

local profile_base = nil

for i,v in ipairs(arg) do
    if v == "--profile" then
	local p = arg[i+1]
	if arg[i+1] then
            profile_base = arg[i+1]
            table.remove(arg, i+1)
        else
            error("--profile requires a filename base")
        end
	table.remove(arg, i)
    end
end




-- lookup functions done in Lua because I haven't protred them to C yet
local function name_scope_search(value, max_depth, scope)
    max_depth = max_depth or 5
    scope = scope or _G
    if max_depth == 0 then
        return false
    end

    for k,v in pairs(scope) do
        if type(v) == type({}) and k ~= "package" and k ~= "_G" and k ~= "__index" then
            local a,b = name_scope_search(value, max_depth-1, v)
            if a then
                return a,k .. "." .. b
            end
        end
        if v == value then
            return true, k
        end
    end
    return false
end


local function fix_mts(x)
    local a,b,c,d = string.find(x, "(.*)%.(.-)&")
    if a then
        return c .. ":" .. d
    end
    return x
end

local function metatable_search(value, max_depth, scope)
    max_depth = max_depth or 5
    scope = scope or _G
    if max_depth == 0 then
        return false
    end


    for k,v in pairs(scope) do
        if k == "metatable" and type(v) == type(print) then
            for k2,v2 in pairs(v()) do
                if value == v2 then
                    return true, k2
                end
            end
        end

        if type(v) == type({}) and k ~= "package" and k ~= "_G" and k ~= "__index" then
            local a,b = metatable_search(value, max_depth-1, v)
            if a then
                return a,k .. "." .. b
            end
        end
    end
    return false
end

-- really local and upvalues
local function name_local_search(value, start_level, source_lookup)
    local level = start_level or 2
    source_lookup = source_lookup or true

    local env = debug.getinfo(level)

    while env  do
        local i,k,v = 1,debug.getlocal(level, 1)
        local pos = env.short_src .. ":" .. env.currentline
        while k do
            if value == v then
                return true, k, pos
            end
            if type(v) == type({}) then
                local a,b,c = name_scope_search(value, 5, v)
                if a then
                    return true, k .. "." .. b, pos
                end
            end
            i,k,v = i+1,debug.getlocal(level, i+1)
        end


        if env.func then
            local i,k,v = 1,debug.getupvalue(env.func, 1)
            while k do
                if value == v then
                    return true, k, pos
                end
                if type(v) == type({}) then
                    local a,b,c = name_scope_search(value, 5, v)
                    if a then
                        return true, k .. "." .. b, pos
                    end
                end
                i,k,v = i+1,debug.getupvalue(env.func, i+1)
            end
        end

        level = level + 1
        env = debug.getinfo(level)
    end

    return false, name, ""

end

-- given a function value, return a nice string
local function function_lookup(func)
    local name = tostring(func)

    local a,b = name_scope_search(func)
    if a then
        return b
    end

    a,b,c = name_local_search(func, 4)
    if a then
        return b .. "@" .. c
    end

    a,b = metatable_search(func, 4)
    if a then
        return fix_mts(b) -- need to check here. Not seeing A:f() 
    end

    local env = debug.getinfo(func)

    if env.what == "Lua" then
        local file  = env.short_src
        local line  = env.linedefined

        if file and line then
            return "Function at " .. file .. ":" .. line
        end
        return "Unknown function " .. name
    end

    if env.what == "C" then
        return "C " .. name
    end

    if env.what == "main" then
        return "main chunk"
    end

    return env.what
end




if profile_base then
    local tn = os.tmpname()
    os.execute("/bin/date > " .. tn)
    local f = io.open(tn, "r")
    local date = f:read("*l") or "Unknown date"
    f:close()
    os.execute("rm -f " .. tn)

    _set_profile_data({profile_base, date})

    _profile_set_lookup(function_lookup)
    _profile_start()
end

_set_profile_data = nil
_profile_start = nil
_profile_set_lookup = nil

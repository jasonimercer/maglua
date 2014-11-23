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
local function_lookup_list = {}
local function function_lookup(func)
    local name = tostring(func)
    if function_lookup_list[name] then
        return function_lookup_list[name]
    end

    local function add(s)
        function_lookup_list[name] = s
        return s
    end

    local a,b = name_scope_search(func)
    if a then
        return add(b)
    end

    a,b,c = name_local_search(func, 4)
    if a then
        return add(b .. "@" .. c)
    end

    a,b = metatable_search(func, 4)
    if a then
        return add(fix_mts(b))
    end

    local env = debug.getinfo(func)

    if env.what == "Lua" then
        local file  = env.short_src
        local line  = env.linedefined

        if file and line then
            return add("Function at " .. file .. ":" .. line)
        end
        return add("Unknown function " .. name)
    end

    if env.what == "C" then
        return add("C " .. name)
    end

    if env.what == "main" then
        return add("main chunk")
    end

    return add(env.what)
end

local call_data = {}
local stack = {}
local stack_size = 0

-- call
-- return
-- tail return

local timer_all = 1
local timer_func = 2
local call_count = 3
local recursion_level = 4
local total_inclusive = 5
local total_exclusive = 6
local id_pos = 7
local name_pos = 8

local profile_total_timer = Timer.new()


local function new_call_data(id, name)
    return {0, 0, 0, 0, 0, 0, id, name}
end

local function hook(event, lineno)

    local ptt_e = profile_total_timer:elapsed()

    local t = debug.getinfo(2)
    local name, id = function_lookup(t.func), tostring(t.func)

    if event == "call" then
        local sss = stack[stack_size]
        if sss and call_data[sss] then
            local cd = call_data[sss]
            cd[total_exclusive] = cd[total_exclusive] + (ptt_e - cd[timer_func])
        end

        stack_size = stack_size + 1
        stack[stack_size] = id

        if call_data[id] == nil then
            call_data[id] = new_call_data(id, name)
        end

        local t = call_data[id]

        t[call_count] = t[call_count] + 1
        t[recursion_level] = t[recursion_level] + 1
        if t[recursion_level] == 1 then
            local ptt = profile_total_timer:elapsed()
            t[timer_all] = ptt_e
            t[timer_func] = ptt_e
        end
    end

    if event == "return" or event == "tail return" then
        local id = stack[stack_size]
        stack[stack_size] = nil
        stack_size = stack_size - 1

        local t = call_data[id]
        if t then
            t[recursion_level] = t[recursion_level] - 1
            if t[recursion_level] == 0 then
                t[total_inclusive] = t[total_inclusive] + (ptt_e - t[timer_all])
            end
            t[total_exclusive] = t[total_exclusive] + (ptt_e - t[timer_func])
        end

        id =  stack[stack_size]
        local t = call_data[id]
        if t then
            t[timer_func] = ptt_e
        end
    end
end



if profile_base then
    local tn = os.tmpname()
    os.execute("/bin/date > " .. tn)
    local f = io.open(tn, "r")
    local date = f:read("*l")
    f:close()
    os.execute("rm -f " .. tn)

    profile_total_timer:start()

    _set_profile_data({call_data, profile_total_timer, profile_base, date})

    debug.sethook(hook, "cr", 0)
end

_set_profile_data = nil



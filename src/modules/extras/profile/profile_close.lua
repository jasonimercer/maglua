debug.sethook()
local x = _get_profile_data() or {}

local timer_all = 1
local timer_func = 2
local call_count = 3
local recursion_level = 4
local total_inclusive = 5
local total_exclusive = 6
local id_pos = 7
local name_pos = 8


local call_data = x[1]
local profile_total_timer = x[2]
local profile_base = x[3]


if call_data then -- we were profiling
    profile_total_timer:stop()
    profile_total_timer = profile_total_timer:elapsed()

    local cd = {}
    for k,v in pairs(call_data) do
        table.insert(cd, v)
    end

    local function cd_sort_inclusive(a,b)
        return a[total_inclusive] > b[total_inclusive]
    end
    local function cd_sort_exclusive(a,b)
        return a[total_exclusive] > b[total_exclusive]
    end

    local result_table = {}

    table.sort(cd, cd_sort_inclusive)



    local fcalls = 0
    for i=1,table.maxn(cd) do
        local c = cd[i]
        
        if c[recursion_level] == 0 then
            local time_inclusive = c[total_inclusive]
            local time_exclusive = c[total_exclusive]
            local percent_in = 100 *time_inclusive / profile_total_timer
            local percent_ex = 100 *time_exclusive / profile_total_timer
            if c[name_pos] ~= "(*temporary)@[C]:-1" and
                string.sub(c[name_pos],1,15) ~= "(for generator)" then
                table.insert(result_table, string.format("%-50s %7d       %6.2f%%   %6.2f%%", c[name_pos], c[call_count], percent_in, percent_ex))
                fcalls = fcalls + c[call_count]
            end
        end
    end

    local fn = profile_base .. "_" .. string.format("%06d", os.pid()) .. ".txt"
    local f = io.open(fn, "w")

    f:write("Profile data for " .. profile_base .. ", process id: " .. string.format("%06d", os.pid()) .. "\n")
    f:write(fcalls .. " function calls.\n")
    f:write(string.format("%8g second runtime.\n", profile_total_timer ))
    f:write("\n")
    f:write("                                                                 Percent Runtime\n")
    f:write("Function Name                               Function Calls     Inclusive Exclusive\n")
    f:write(string.rep("=", 82) .. "\n")
    f:write(table.concat(result_table, "\n") .. "\n")
    f:close()

    io.stderr:write("Profile written to `" .. fn .. "'\n")
end


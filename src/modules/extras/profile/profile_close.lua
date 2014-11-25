debug.sethook()
local x = _get_profile_data()

if x then -- we were profiling
    local result_table = {}

    profile_start_date = x[2]
    profile_base = x[1]

    local call_data, profile_total_time = _profile_support_stop()

    -- it is possible to have many functions with the same name
    -- this can happen when you create a temporary local function
    -- inside another function that gets called several times
    -- so what we will do is create a looking on names mapping
    -- to tables of function keys
    local fname_mapping = {}
    for id,cd in pairs(call_data) do
        fname_mapping[cd.name] = fname_mapping[cd.name] or {}
        table.insert(fname_mapping[cd.name], id)
    end

    -- now let's have a table of fnames so we can sort
    local fnames = {}
    for k,v in pairs(fname_mapping) do
        table.insert(fnames, k)
    end

    local function total_x(name, x)
        local sum = 0
        for k,id in pairs(fname_mapping[name]) do
            sum = sum + call_data[id][x]
        end
        return sum
    end

    local function cd_sort_inclusive(a,b)
        return total_x(a, "inclusive") > total_x(b, "inclusive")
    end
    local function cd_sort_exclusive(a,b)
        return total_x(a, "exclusive") > total_x(b, "exclusive")
    end


    table.sort(fnames, cd_sort_inclusive)

    local fcalls = 0
    for i=1,table.maxn(fnames) do
        local name = fnames[i]
        if total_x(name, "recursion_level") == 0 then
            local time_inclusive = total_x(name, "inclusive")
            local time_exclusive = total_x(name, "exclusive")
            local call_count = total_x(name, "call_count")
            local percent_in = 100 *time_inclusive / profile_total_time
            local percent_ex = 100 *time_exclusive / profile_total_time
            --if not string.find(name, "^%(%*temporary%)") and not string.find(name, "^%(for generator%)") then

            local name_len = string.len(name)
            local cc_len = string.len(tostring(call_count))

            local p1 = name .. string.rep(" ", 58 - name_len - cc_len) .. call_count

                table.insert(result_table, 
                             string.format(p1 .. "       %6.2f%%   %6.2f%%", percent_in, percent_ex))
                fcalls = fcalls + call_count
            --end
        end
    end

    local fn_base = profile_base .. "_" .. string.format("%05d", os.pid())
    local fn = fn_base .. ".txt"
    local f = io.open(fn, "w")

    f:write("MagLua Profile Report\n\n")

    local cmd = tostring(command_line)
    local date = tostring(profile_start_date)
    local host = _profile_gethostname()
    local pid = string.format("%05d", _profile_getpid())
    local time = string.format("%6g", profile_total_time)
    -- fcalls

    f:write(string.format("  Command Line: %s\n", cmd))
    f:write(string.format("     Host Name: %-30s        PID: %-20s\n", host, pid))
    f:write(string.format("    Start Date: %-30s Total Time: %-20s\n", date, time .. " seconds"))
    f:write(string.format("Function Calls: %s\n", fcalls))

    f:write("\n")
    f:write("                                                                 Percent Runtime\n")
    f:write("Function Name                               Function Calls     Inclusive Exclusive\n")
    f:write(string.rep("=", 82) .. "\n")
    f:write(table.concat(result_table, "\n") .. "\n")

    local num_names = table.maxn(fnames)
    local function name_idx(name)
        for i=1,num_names do
            if fnames[i] == name then
                return i
            end
        end
    end

    -- find function calls
    local function name2name(src, dest)
        local sum = 0

        local src_idxs = fname_mapping[src]
        local dest_idxs = fname_mapping[dest]

        for k1,sidx in pairs(src_idxs) do
            for k2,didx in pairs(dest_idxs) do 
                sum = sum + (call_data[sidx].calls[didx] or 0)
            end
        end

        return sum
    end


    f:write("\nCalling Relationship between functions:\n")
    f:write("Source Function Name         Function Calls   Called Function\n")
    f:write(string.rep("=", 110) .. "\n")

    local function percentages(name)
        local time_inclusive = total_x(name, "inclusive")
        local time_exclusive = total_x(name, "exclusive")
        local percent_in = 100 *time_inclusive / profile_total_time
        local percent_ex = 100 *time_exclusive / profile_total_time

        return string.format("[%6.2f%%:%6.2f%%]", percent_in, percent_ex)
    end


    for i=1,num_names do
        local sname = fnames[i]
        local written = 0
        for j=1,num_names do
            local dname = fnames[j]
            local num_calls = name2name(sname, dname)
            if num_calls > 0 then
                local prefix = ""
                if written == 0 then
                    written = 1
                    local cc = "(" .. total_x(sname, "call_count") .. ")"
                    local len1 = string.len(sname)
                    local len2 = string.len(cc)

                    f:write( "\n" .. sname .. string.rep(" ", 44-len1-len2) .. cc .. "\n")
                    prefix = percentages(sname)
                end

                local time_inclusive = total_x(dname, "inclusive")
                local time_exclusive = total_x(dname, "exclusive")
                local call_count = total_x(dname, "call_count")
                local percent_in = 100 *time_inclusive / profile_total_time
                local percent_ex = 100 *time_exclusive / profile_total_time

                f:write(string.format("%-35s%8g    %40s %s(%g)\n", prefix, num_calls, dname, percentages(dname), call_count))
            end
        end
    end



    f:close()
    io.stderr:write("Profile report written to `" .. fn .. "'\n")


    if false then
    f = io.open(fn_base .. ".dot", "w")

    f:write("digraph call_graph {\n rankdir=LR;\n")


    for i=1,num_names do
        local name = fnames[i]

        --127 [label=Lee, width="0.22222", height="0.15278", group=11, fontsize=7, pos="570.02,1089.4"];

        f:write(string.format([[    node_%d [label="%s\n%d %5.2f%%", shape=rectangle ] ]] .. "\n", i, name,  total_x(name, "call_count"), total_x(name, "exclusive")))

    end


    for i=1,num_names do
        local sname = fnames[i]
        for j=1,num_names do
            local dname = fnames[j]
            local num_calls = name2name(sname, dname)
            if num_calls > 0 then
                f:write(string.format([[    node_%d -> node_%d [ label = "%d" ] ]] .. "\n", i, j, num_calls))
            end
        end
    end
    
    f:write("}\n")

    f:close()
    io.stderr:write("Profile call graph written to `" .. fn_base .. ".dot'\n")
    end

end

_profile_support_stop = nil
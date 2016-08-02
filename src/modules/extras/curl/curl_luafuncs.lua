-- CUrl

local orig_dofile = dofile

function dofile(x)
    local a, b = curl(x)
    
    if a == false then -- try the default?
        return orig_dofile(x)
    end
    
    return assert(loadstring(a, "=" .. x))()
end

local orig_io_open = io.open

local function ioopen(filename, mode)
    mode = mode or "r"
    
    if mode == "r" then
        local a, b = curl(filename)

        if a == false then
            return orig_io_open(filename, mode)
        end

        local f = io.tmpfile()

        f:write(a)
        f:seek("set", 0)
        return f
    end

    return orig_io_open(filename, mode)
end

io.open = ioopen
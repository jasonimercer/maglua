local MODNAME = "WebServer"
local MODTAB = _G[MODNAME]
local mt = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

local methods = {}
local functions = {}

-- initialize/set/get internal data
-- "dd" means nothing, could read it as "do data"
local function dd(ws, new_data)
    if new_data then
        ws:_setinternaldata(new_data)
    end
    local t = ws:_getinternaldata() or {}
    return t
end


methods["setPort"] = {
    "Set the port that the webserver will listen on.",
    "1 Integer: If you are running as non-root, it must be above 1024.",
    "",
    function(ws, port)
        ws:_setPort(port)
    end
}
methods["port"] = {
    "Set the port that the webserver will listen on.",
    "",
    "1 Integer: Port the webserver will listen on",
    function(ws)
        return ws:_port()
    end
}


methods["addPage"] = {
    "Add a page to the WebServer",
    "1 String, 1 String or 1 Function or Nil: The first string is the path to the file, for example \"/\" or \"/test/index.html\". The second argument represents the contents at that page. If it is a string then it's considered static data and will be presented to the web browser. If it is a function, that function will be called with the query string encoded as a table of key-value pairs. This function is expected to return an integer representing an http status code (200 is OK) and a  string which will be presented to the web browser. If a nil is supplied as content then that page will be removed.",
    "",
    function(ws, path, page)
        local data = dd(ws)
        data.pages = data.pages or {}
        data.pages[path] = page
        dd(ws, data)
    end
}

methods["files"] = {
    "Get all the files registered with the WebServer",
    "",
    "1 Table of files: Each key will be the path and the value will be the contents, either a string or a function.",
    function(ws, path, page)
        local t = {}
        local data = dd(ws)
        data.pages = data.pages or {}
        for k,v in pairs(data.pages) do
            t[k] = v
        end
        dd(ws, data)
        return t
    end
}

local function unescape(s)
    s = string.gsub(s, "+", " ")
    s = string.gsub(s, "%%(%x%x)", function (h)
                                       return string.char(tonumber(h, 16))
                                   end)
    return s
end

functions["unescape"] = {
"Convert URL extended character codes into characters",
"1 String: Text with URL extended characters.",
"1 String: Plain text form of URL escaped text.",
unescape
}


local http_status = {}
http_status[200] = "OK"
http_status[300] = "Multiple Choices"
http_status[301] = "Moved Permanently"
http_status[302] = "Found"
http_status[304] = "Not Modified"
http_status[307] = "Temporary Redirect"
http_status[400] = "Bad Request"
http_status[401] = "Unauthorized"
http_status[403] = "Forbidden"
http_status[404] = "Not Found"
http_status[410] = "Gone"
http_status[500] = "Internal Server Error"
http_status[501] = "Not Implemented"
http_status[503] = "Service Unavailable"
http_status[550] = "Permission denied"


function mysplit(inputstr, sep)
    if sep == nil then
        sep = "\n"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

local function decode_request(message)
    message = mysplit(message)

    local a,b,verb,path,protocol = string.find(message[1], "^(%S+)%s+(%S+)%s+(%S+)")

    local cgi = {}
    if path then
        local a,b,p,q = string.find(path, "(.-)%?(.+)")

        if a then
            path = p
            for name, value in string.gfind(q, "([^&=]+)=([^&=]+)") do
                name = unescape(name)
                value = unescape(value)
                cgi[name] = value
            end
        end
    end

    local d = {}
    d.verb = verb
    d.path = path
    d.protocol = protocol
    d.query = cgi


    for i=2,table.maxn(message) do
        local a,b,key,value = string.find(message[i], "^%s*(.-):%s*(.*)$")
        if a then
            d[string.lower(key)] = value
        end
    end

    return d
end


-- todo add path, server, port to args
local function html_print(write, message, code)

    code = code or 200
    local status = http_status[code]

    local header = string.format([[HTTP/1.1 %d %s
Server: MagLua/%s
Content-Type: text/html; charset=UTF-8
Content-Length: %d
Accept-Ranges: bytes
Connection: close

]], code, status, version(), string.len(message))

    local content = header .. message

    write(content)
end

local function make_stat_page(stat, request)
    if stat == 404 then
        return string.format([[<html><head>
<title>404 Not Found</title>
</head><body>
<h1>Not Found</h1>
<p>The requested URL %s was not found on this server.</p>
<hr>
<address>MagLua/%d Server at %s</address>
</body></html>
]], request.path or "", version(), request.host or "")
    end

    if http_status[stat] then
        local nn = tostring(stat) .. " " .. http_status[stat]
        return string.format([[<html><head>
<title>%s</title>
</head><body>
<h1>%s</h1>
<hr>
<p>The requested URL %s returned %s.</p>
<address>MagLua/%d Server at %s</address>
</body></html>
                             ]], nn, stat, version(), request.path or "", http_status[stat], request.host or "")
    end

    return string.format([[<html><head>
<title>Server Error %s</title>
</head><body>
<h1>Error %s</h1>
<hr>
<p>The requested URL %s returned %s.</p>
<address>MagLua/%d Server at %s</address>
</body></html>]], stat, stat, request.path or "", version(), request.host or "")
end


function client_func(ws, session_id, message, write)
    local request = decode_request(message)

    local data = dd(ws)
    data.pages = data.pages or {}

    local page = data.pages[request.path]

    if page == nil then
        html_print(write, make_stat_page(404, request), 404)
    else
        if type(page) == type(type) then
            local stat, content = page(request.query or {})
            if stat == 200 then
                html_print(write, content)
            else
                html_print(write, make_stat_page(stat, request),  stat)
            end
        else
            html_print(write, page, stat)
        end
    end

    -- print(session_id, message)
end

methods["start"] = {
    "Start the WebServer",
    "",
    "",
    function(ws)
        ws:_setClientFunction(client_func)
        ws:_start()
    end
}


-- inject above into existing metatable for WebServer object
for k,v in pairs(methods) do
    mt[k] = v[4]
end

-- inject functions
for k,v in pairs(functions) do
    MODTAB[k] = v[4]
end

-- backup old help function for fallback
local help = MODTAB.help

-- create new help function for methods above
MODTAB.help = function(x)
                  for k,v in pairs(methods) do
                      if x == v[4] then
                          return v[1], v[2], v[3]
                      end
                  end

                  for k,v in pairs(functions) do
                      if x == v[4] then
                          return v[1], v[2], v[3]
                      end
                  end

                  -- fallback to old help if a case is not handled
                  if x == nil then -- VecCS overview
                      return "Custom MagLua Webserver object","",""
                  end

                  return help(x)
              end


--
-- This script combines all given files into a manual
-- 
-- call with
--    maglua GenerateManual.lua [list of manual files]


function processfile(filename)
	local a, b, c = string.find(filename, "(.*)%.lua$")
	local f = {}
	if a == nil then
		return
	end
	f["SourceFile"] = filename
	f["SourceFileBase"] = c
	
	local g = io.open(filename, "r")
	local line = g:read("*line")
	
	a, b, c = string.find(line, "%-%-%s*Section%:%s*(.-)%s*$")
	
	if a == nil then
		return
	end
	
	f["SectionName"] = c
	
	local buffer = ""
	local i = 1
	local intext = true
	for w in string.gfind(g:read("*all"), "(.-\n)") do
		if string.find(w, "^%-%-") then --comment -> text
			if intext then --add to buffer
				buffer = buffer .. w
			else -- new text
				f[i] = {"code", buffer}
				buffer = w
				i = i + 1
				intext = true
			end
		else -- code
			if intext then -- new 
				f[i] = {"text", buffer}
				buffer = w
				i = i + 1
				intext = false
			else
				buffer = buffer .. w
			end
		end
	end

	if intext and buffer ~= "" then -- new 
		f[i] = {"text", buffer}
		i = i + 1
	end
	if not intext and buffer ~= "" then -- new 
		f[i] = {"code", buffer}
		i = i + 1
	end

	f["Count"] = i-1

	f["Output"] = outputbuffer

	return f
end


function writeheader(f, name)
	f:write("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2//EN\">\n")
	f:write("<html><head><title>" .. name .. "</title>\n")
	f:write("<link rel=\"stylesheet\" type=\"text/css\" href=\"maglua.css\">\n")
	f:write("</head>\n")
	f:write("<body bgcolor=\"white\">\n")
end

function writefooter(f, desc)
	f:write("</body>\n</html>\n")
end


function trimws(text)
	local a, b, c = string.find(text, "^%s*(.-)%s*$")
	if a then
		return c
	end
	return text
end

function openFile(desc)
	local f = io.open(desc["SourceFileBase"] .. ".html", "w")
	return f
end

function writeText(f, text)
	for line in string.gfind(text, "(.-)\n") do
		local a, b, c = string.find(line, "^%-%-%s*(.*)$")
		if a then
			if c == "" then
				f:write("<p>")
			else
				f:write(c .. "\n")
			end
		end
	end
end

function writeCode(f, text)
	f:write("\n<pre>\n")
	for line in string.gfind(trimws(text) .. "\n", "(.-\n)") do
		f:write(line)
	end
	f:write("</pre>\n\n")
end

function doCode(text)
	local outputbuffer = {}
	function bprint(...)
		local r = ""
		for i,v in ipairs(arg) do
			r = r .. tostring(v) .. "\t"
		end
		table.insert(outputbuffer, r .. "\n")
	end

	local p = print
	print = bprint

	assert(loadstring(text))()
	
	print = p
	return outputbuffer
end

local _all = {}
for k,v in pairs(arg) do
	local desc = processfile(v)

	if desc then
		print("Processing `" .. desc["SourceFile"] .. "'")

		local f = openFile(desc)
		writeheader(f, desc["SectionName"])

		f:write("<h2>" .. desc["SectionName"] .. "</h2>\n")

		for i=1,desc["Count"] do
			if desc[i][1] == "text" then
				writeText(f, desc[i][2])
			end
			if desc[i][1] == "code" then
				writeCode(f, desc[i][2])

				local res = doCode(desc[i][2])
				if res[1] then
					local buf = ""
					for a,b in pairs(res) do
						buf = buf .. b
					end
					f:write("<hr>\n")
					writeCode(f, buf)
				end
			end
		end

		writefooter(f, desc)
		f:close()
		print("     Wrote `" .. desc["SourceFileBase"] .. ".html'")
		table.insert(_all, desc)
	end
end

print("Generating `index.html'")

--now to write the index
f = io.open("index.html", "w")
writeheader(f, "Programming With MagLua")
f:write("<h1>Programming With MagLua</h1>\n")
f:write("<p>These examples will demonstrate how the elements of MagLua may be combined to\n")
f:write(   "create full micromagnetic simulations.\n")
f:write("<hr>\n")
f:write("<h2>Contents</h2>\n")
for k,desc in pairs(_all) do
	f:write("<a href=\"" .. desc["SourceFileBase"] .. ".html\">" .. desc["SectionName"] .. "</a><br>")
end
writefooter(f)
f:close()

print("     Wrote `index.html'")


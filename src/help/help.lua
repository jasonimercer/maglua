--
-- This script uses the built in help features of maglua to build an html help file
-- If an argument is supplied, that will be the output filename
--

filename = arg[1] or "maglua.html"
f = io.open(filename, "w")

function lp(txt) -- Link Process, change *TEXT* into <a href="#TEXT">TEXT</a>
	local a, b, c, d, e = string.find(txt, "^(.*)\*(.-)\*(.*)$")

	if a then
		return lp(c .. "<a href=\"#" .. d .. "\">" .. d .. "</a>" .. e)
	end
	return txt
end

f:write([[
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<style media="screen" type="text/css">
body {
	color: #000000 ;
	background-color: #FFFFFF ;
	font-family: sans-serif ;
	text-align: justify ;
	margin-right: 20px ;
	margin-left: 20px ;
}

h1, h2, h3, h4 {
	font-weight: normal ;
	font-style: italic ;
}

h2 {
	padding-top: 0.4em ;
	padding-bottom: 0.4em ;
	padding-left: 20px ;
	margin-left: -20px ;
	background-color: #E0E0FF ;
}

h3 {
	padding-left: 8px ;
	border-left: solid #E0E0FF 1em ;
}

table h3 {
	padding-left: 0px ;
	border-left: none ; 
}

a:link {
	color: #000080 ;
	background-color: inherit ;
	text-decoration: none ;
}

a:visited {
	background-color: inherit ;
	text-decoration: none ;
}

a:link:hover, a:visited:hover {
	color: #000080 ;
	background-color: #E0E0FF ;
}

a:link:active, a:visited:active {
	color: #FF0000 ;
}

hr {
	border: 0 ;
	height: 1px ;
	color: #a0a0a0 ;
	background-color: #a0a0a0 ;
}

:target {
	background-color: #F8F8F8 ;
	padding: 8px ;
	border: solid #a0a0a0 2px ;
}
</style>

<title>MagLua Reference Manual</title>
<META HTTP-EQUIV="content-type" CONTENT="text/html; charset=iso-8859-1">
</head>
<body>

<H1>MagLua</H1>
<p>MagLua is an extension to the base Lua language that allows a user to build micromagnetic simulations in the Lua scripting language.
<p>MagLua is composed of 3 conceptual parts
<ul>
	<li>Data - Spin vectors and fields, these are held in a *SpinSystem*.
	<li>Operators - Objects which calculate fields based on spins or external influences such as *Anisotropy*, *Dipole*, *Thermal*, etc.
	<li>Integrators - Objects which update spin orientations based on calculated effective fields. Different integrators can be created using *LLG*.
</ul>
<p>The following is a list of the objects and functions which may be combined to create a simulation.]])

-- Add a section heading
function addsection(name, level, effect, noadd)
	if noadd ~= true then
		table.insert(index, {name, level})
	end

	if effect then
		f:write("<p>\n<h" .. level .. "><a name=\"" .. name .. "\"><" .. effect .. ">" .. name .. "</" .. effect .. "></a></h" .. level .. ">\n")
	else
		f:write("<p>\n<h" .. level .. "><a name=\"" .. name .. "\">" .. name .. "</a></h" .. level .. ">\n")
	end
end

-- write desc, input and output of a function/method
function dl(a, b, c)
	f:write("<dl>\n")
	if a ~= "" then f:write("<dt>Description</dt><dd>" .. lp(a) .. "</dd>\n") end
	if b ~= "" then f:write("<dt>Input</dt><dd>" .. lp(b) .. "</dd>\n") end
	if c ~= "" then f:write("<dt>Output</dt><dd>" .. lp(c) .. "</dd>\n") end
	f:write("</dl><hr>\n")
end

-- table for the index
index = {}


-- sorted list of all tables (with help function) in scope
s = {}
for k,v in pairs(_G) do
	if type(v) == "table" then
		if type(v.help) == "function" then
			table.insert(s, {k, v})
		end
		for x,y in pairs(v) do
			if type(y) == "table" and k ~= "_G" then
				if type(y.help) == "function" then
					table.insert(s, {k .. "." .. x, y})
				end
			end
		end
	end
end

-- sort tables besed on first element (key)
function comp(a, b)
	return a[1] < b[1]
end

table.sort(s, comp)

-- over all sorted tables
for i,v in ipairs(s) do
	t = v[2]
	n = v[1]
	a, b, c = t.help()
	if a then
		addsection(n, 2)
		f:write("<p>\n" .. lp(a) .. "\n")
		if b then
			f:write("<p><dl><dt>" .. n .. ".new() takes the following arguments</dt><dd>" .. lp(b) .. "</dd></dl>\n")
		end
		f:write("<hr>\n");
	end

	-- write documentation about all functions
	local xx = {}
	for k,v in pairs(t) do
		a, b, c = t.help(v)
		if a then
			table.insert(xx, {k, a, b, c})
		end
	end
	table.sort(xx, comp)
	for k,v in ipairs(xx) do
		addsection(n .. "." .. v[1], 3)
		dl(v[2], v[3], v[4])
	end

	-- write documentation about all methods
	local yy = {}
	for k,v in pairs(t.metatable()) do
		if k ~= "__index" then
			a, b, c = t.help(v)
			if a then
				table.insert(yy, {k, a, b, c})
			end
		end
	end
	table.sort(yy, comp)
	for k,v in ipairs(yy) do
		addsection(n .. ":" .. v[1], 3, "code")
		dl(v[2], v[3], v[4])
	end
end

-- here we will explain what a 3Vector is, it appears in the documentation
addsection("3Vector", 2, nil, true) --true - don't add index to index
dl("A 3Vector is an argument of some methods in MagLua, it can either be 3 numbers or a table with 3 values. If it is a table with less than 3 values, sensible defaults are used or an error is returned if none exist.", "", "") 

-- write index, 4 columns
addsection("Index", 2, nil, true) --true - don't add index to index
f:write("<table width=\"100%\">\n")
f:write("<tr align=\"top\">\n")
f:write("<td valign=\"top\">\n")

elements = table.maxn(index)
initrowcount = elements/4
rowcount = initrowcount

for k,v in pairs(index) do
	a, b = v[1], v[2]
	if rowcount < 0  then --only break columns on new sections
		f:write("</td>\n<td valign=\"top\">\n")
		--rowcount = rowcount + initrowcount
		rowcount = initrowcount
	end

	if b == 2 then
		f:write("<p>\n")
	end

	f:write("<a href=\"#"..a.."\">"..a.."</a><br>\n")
	rowcount = rowcount - 1
end

f:write("</td></tr></table><hr>\n")


f:write("<table>\n")
local i = 0

for w in string.gfind(info(), "(.-)\n") do
	local a, b, c, d = string.find(w, "(.-)%s*%:(.*)")
	if a then
		f:write("<tr><td width=15%>" .. c .. ":<td>" .. d .. "\n")
	end
end
f:write("</table>\n<hr>")

f:write("</body>\n</html>\n")
f:close()

print("Wrote HTML help to `" ..filename .. "'")

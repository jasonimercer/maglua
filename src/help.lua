--
-- This script uses the built in help features of maglua to build an html help file
-- If an argument is supplied, that will be the output filename
--
-- Also provides the default Global Scope help function
-- 

help = help or
function(f)
	if f == nil then --want description for Global Scope
		return "This is the Global Scope for MagLua. The following are custom functions added to the base language to help " ..
			   "create and run simulations." --only returning 1 string
	end
end


function write_help(file_handle, optional_links)
	local f = file_handle or io.stdout
	optional_links = optional_links or {}
	
	local function nq(x) --nq = noquotes
		return string.gsub(x, "\"", "")
	end

	function lp(txt) -- Link Process, change *TEXT* into <a href="#TEXT">TEXT</a>. Also changing \n for <br>\n
-- 		txt = string.gsub(txt, "<", "&lt;")
-- 		txt = string.gsub(txt, ">", "&gt;")
		
		
		
		local function _ws(txt) --tab, br
			txt = string.gsub(txt, "\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
			txt = string.gsub(txt, "\n", "<br>")
			return txt
		end
			
		local function _lp(txt)
			local a, b, c, d, e = string.find(txt, "^(.*)\*(.-)\*(.*)$")

			if a then
				return _lp(c .. "<a href=\"#" .. nq(d) .. "\">" .. d .. "</a>" .. e)
			end
			return txt
		end
		
		--don't want to process <pre>xyz</pre>
		local function parts(txt, t)
			t=t or {}
			local a, b, c, d, e = string.find(txt, "^(.-)(<pre>.-</pre>)(.*)")
			if a then
				table.insert(t, {"txt", c})
				table.insert(t, {"pre", d})
				return parts(e, t)
			end
			table.insert(t, {"txt", txt})
			return t
		end
		local p = parts(txt)
		
		for k,v in pairs(p) do
			if v[1] == "txt" then
				p[k][2] = _lp(_ws(p[k][2]))
			end
		end
		
		local s = {}
		for k,v in pairs(p) do
			s[k] = v[2]
		end
		return table.concat(s, " ")
	end

	function write_optional_links()
		local data = {}

		if optional_links["devstats"] then
			table.insert(data, 
					string.format("<p>Development statistics can be found at the StatSVN <A HREF=\"%s\">page</A>.", optional_links["devstats"]))
		end
		
		if optional_links["tips"] then
			table.insert(data, 
					string.format("<p>Examples of specific MagLua functions and objects can be found at the MagLua Tips and Tricks <A HREF=\"%s\">page</A>.", optional_links["tips"]))
		end
		
		return table.concat(data, " ")
	end
	
	f:write([[
	<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
	<html>
	<head>
	<style media="screen,print" type="text/css">
	body {
		color: #000000 ;
		background-color: #FFFFFF ;
		font-family: sans-serif ;
		text-align: justify ;
		margin-right: 40px ;
		margin-left: 40px ;
	}

	pre {
		-webkit-print-color-adjust: exact; 
		padding: 5px;
		background-color: #eeffcc;
		color: #333333;
		line-height: 120%;
		border: 1px solid #ac9;
		border-left: none;
		border-right: none;
	}


	h1,h2
	{
		-webkit-print-color-adjust: exact; 
		color: #FFFFFF;
		font-weight: normal;
		padding-top: 0.4em ;
		padding-bottom: 0.4em ;
		padding-left: 20px ;
		margin: 20px -20px 10px -20px;
		background-color:  #20435c;
	}


	h3,h4,h5,h6
	{
		-webkit-print-color-adjust: exact; 
		font-family: sans-serif;
		background-color: #f2f2f2;
		font-weight:bold;
		color: #20435c;
		border-top: 1px solid #999;
		margin: 20px -20px 10px -20px;
		padding: 3px 0 3px 10px;
		display: block;
	}

	a:link {
		color: #20435c ;
		background-color: inherit ;
		text-decoration: none ;
	}

	a:visited {
		color: #20435c ;
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
		margin: 20px -20px 10px -20px;
		padding: 3px 0 3px 10px;
	}
	dt 
	{
		font-weight:bold;
	}
	
	h1:target,h2:target {
		background-color: #3c7dab ;
		border: solid #4fa5e2 1px ;
	}
	</style>

	<title>MagLua Reference Manual</title>
	<META HTTP-EQUIV="content-type" CONTENT="text/html; charset=iso-8859-1">
	</head>
	<body>
	<table width=\"100%\"><tr><td> <!-- for rendering to pdf -->
	]])

	f:write(lp([[
<H1>MagLua</H1>
<p>MagLua is an extension to the base Lua language that allows a user to build micromagnetic simulations with the Lua scripting language.	<p>MagLua is composed of 2 conceptual parts following the Data Oriented Design paradigm
<ul><li>Data - Spin vectors and fields, these are held in a *SpinSystem*.<li>Transformations - Objects which modify data. Some calculate fields based on spins or external influences such as *Anisotropy*, *Dipole* and *Thermal*, others update the spin vectors such as *LLG.Cartesian*.	</ul>	<p>The <a href="#Index">Index</a> has links to all objects, methods and functions provided by MagLua.</p>]]))


-- 	print(lp(write_optional_links()))

	f:write(lp(write_optional_links()))

	f:write("\n" .. lp([[<p>The following is a list of the objects and functions which may be combined to create a simulation.]] .. "\n"))
	
	
	-- table for the index
	local index = {}	

	-- Add a section heading
	local function addsection(name, level, effect, noadd)
		if noadd ~= true then
			table.insert(index, {name, level})
		end

		if effect then
			f:write("<p>\n<h" .. level .. "><a name=\"" .. nq(name) .. "\"><" .. effect .. ">" .. name .. "</" .. effect .. "></a></h" .. level .. ">\n")
		else
			f:write("<p>\n<h" .. level .. "><a name=\"" .. nq(name) .. "\">" .. name .. "</a></h" .. level .. ">\n")
		end
	end

	-- write desc, input and output of a function/method
	local function dl(a, b, c)
		a = a or ""
		b = b or ""
		c = c or ""
		f:write("<dl>\n")
		if a ~= "" then f:write("<dt>Description</dt><dd>" .. lp(a) .. "</dd>\n") end
		if b ~= "" then f:write("<dt>Input</dt><dd>" .. lp(b) .. "</dd>\n") end
		if c ~= "" then f:write("<dt>Output</dt><dd>" .. lp(c) .. "</dd>\n") end
		f:write("</dl>\n")
	end




	-- sorted list of all tables (with help function) in scope
	local s = {}
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
	local function comp(a, b)
		return a[1] < b[1]
	end

	table.sort(s, comp)

	-- over all sorted tables
	for i,v in ipairs(s) do
		local t = v[2] --table
		local n = v[1] --name
		local a, b, c = t.help()
		if a then
			addsection(n, 2)
			f:write("<p>\n" .. lp(a) .. "\n")
			if b and b ~= "" then
				f:write(string.format("<H3>%s.new</H3><dl><dt>Constructor Arguments</dt><dd>%s</dd></dl>", n,lp(b)))
-- 				f:write("<p><dl><dt>" .. n .. ".new() takes the following arguments</dt><dd>" .. lp(b) .. "</dd></dl>\n")
			end
-- 			f:write("<hr>\n");
		end

		-- write documentation about all functions
		local xx = {}
		for k,v in pairs(t) do
			local a, b, c = t.help(v)
			if a then
				table.insert(xx, {k, a, b, c})
			end
		end
		table.sort(xx, comp)
		for k,v in ipairs(xx) do
			if n ~= "_G" then
				addsection(n .. "." .. v[1], 3)
			else
				addsection(v[1], 3)
			end
			dl(v[2], v[3], v[4])
		end

		-- write documentation about all methods
		local yy = {}
		if t.metatable ~= nil then
			for k,v in pairs(t.metatable()) do
				if k ~= "__index" then
					a, b, c = t.help(v)
					if a then
						table.insert(yy, {k, a, b, c})
					end
				end
			end
-- 		else
			
		end
		table.sort(yy, comp)
		for k,v in ipairs(yy) do
			addsection(n .. ":" .. v[1], 3, "code")
			dl(v[2], v[3], v[4])
		end
	end


	-- now to process all the dofiles available using the maglua:// protocol
	for k,v in pairs(dofile_get()) do
		local text = dofile_get(k)
		local lines = {}

		while text ~= "" do
			local v1,v2,y,z = string.find(text,"^(.-)\n(.*)$")
			if v1 then
				table.insert(lines, y)
				text = z
			else
				table.insert(lines, a)
				text = ""
			end
		end

		-- going to strip off the first X lines that start with comments
		-- (documentation is done matlab style)

		local docs = {}
		local i = 1
		while lines[i] and string.sub(lines[i], 1,2) == "--" do
			table.insert(docs, string.sub(lines[i], 3,-1))
			i = i + 1
		end

		addsection("dofile(\"maglua://" .. k .. "\")", 2)
		dl(table.concat(docs, "\n"))
	end
	

	-- here we will explain what a 3Vector is, it appears in the documentation
	addsection("3Vector", 2, nil, true) --true - don't add index to index
	dl("A 3Vector is an argument of some methods in MagLua, it can either be 3 numbers or a table with 3 values. If it is a table with less than 3 values, sensible defaults are used or an error is returned if none exist.", "", "") 

	-- write index, 4 columns
	addsection("Index", 2, nil, true) --true - don't add index to index
	f:write("<table width=\"100%\">\n")
	f:write("<tr align=\"top\">\n")
	f:write("<td valign=\"top\">\n")

	local elements = table.maxn(index)
	local initrowcount = elements/4
	local rowcount = initrowcount

	for k,v in pairs(index) do
		local a, b = v[1], v[2]
		if rowcount < 0  then --only break columns on new sections
			f:write("</td>\n<td valign=\"top\">\n")
			--rowcount = rowcount + initrowcount
			rowcount = initrowcount
		end

		if b == 2 then
			f:write("<p>\n")
		end

		f:write("<a href=\"#"..nq(a).."\">"..a.."</a><br>\n")
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
	f:write("</table>\n<br><hr>")

	f:write("</table><!-- for rendering to pdf--></body>\n</html>\n")
end
	
	
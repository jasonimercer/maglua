-- checkpoint_luafuncs.lua
-- 
-- This file provides documentation for the checkpoint functions

dofile("maglua://Help.lua") --get default global scope help function

local h = help

function help(f) -- adding new rule to help function for these cases
	if f == checkpointSave then
		return "Function to save zero or more values to a checkpoint file. \nExample:\n" ..
[[<pre>
function f(x)
	return x*x
end

checkpointSave("checkpoint_help_example.dat", f, 5, {"a", "b"})
</pre>]],
				"1 String, 0 or more Values: The String is the file name to be used, values will be encoded and saved to file",
				""
	end
	
	if f == checkpointLoad then
		return "Function to load zero or more values from a checkpoint file. \nExample:\n" ..
[[<pre>
g, value, t = checkpointLoad("checkpoint_help_example.dat")
print(g(value), t[1], t2])
</pre>
]],
				"1 String: The String is the file to be read",
				"0 or more Values: The values encoded in the file are returned from the function."
	end
	
-- 	These functions will save and load data using the uuencode/uudecode algorithm.
-- 
			
	if f == checkpointToString then
		return "Function to save zero or more values to a checkpoint string encoded using the uuencode algorithm. " ..
			   "Encoded data use only printable characters and so can be easily emailed, copied to a clipboard, " ..
			   "stored in a text file or stored in a database.\n" ..
			   "Example:" ..
[[<pre>
function f(x)
	return x*x
end

s = checkpointToString(f, 5, {"a", "b"})
print(s)
</pre>
Expected Output:
<pre>
begin 600 checkpoint.dat
M0TA%0TM03TE.5```````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M```````````````````````````````````````````````````#````&lt;P``
M``8```!K````&TQU85$``00(!`@`"0````````!`86%A+FQU80`!`````P``
M```!``(#````3@```%X```$&gt;`(`````````````#`````@````(````#````
M`0````(`````````&gt;````````@`````````,`````P```````````!1`-```
M``4````"`````P```````````/`_!`````(```!A``,`````````````0`0`
(```"````8@``
`
end
</pre>
]],
				"0 or more Values: Values will be encoded and returned as a string",
				"1 String: The string representing the values as a printable string encoded using the uuencode algorithm."
	end
	
	
		if f == checkpointFromString then
		return "Function to load zero or more values from a checkpoint string encoded using the uuencode algorithm. " ..
			   "Encoded data use only printable characters and so can be easily emailed, copied to a clipboard, " ..
			   "stored in a text file or stored in a database.\n" ..
			   "Example:" ..
[[
<pre>
s = ]] .. "[[" .. [[begin 600 checkpoint.dat
M0TA%0TM03TE.5```````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M```````````````````````````````````````````````````#````&lt;P``
M``8```!K````&TQU85$``00(!`@`"0````````!`86%A+FQU80`!`````P``
M```!``(#````3@```%X```$&gt;`(`````````````#`````@````(````#````
M`0````(`````````&gt;````````@`````````,`````P```````````!1`-```
M``4````"`````P```````````/`_!`````(```!A``,`````````````0`0`
(```"````8@``
`
end]] .. "]]\n" .. [[
	
f, v, t = checkpointFromString(s)

</pre>
]],
				"1 String: The string representing the values as a printable string encoded using the uuencode algorithm.",
				"0 or more Values: Values will be decoded and returned."
	end
	
	
	return h(f) -- calling previous help function
end

-- Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
--
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
-- IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
-- CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
-- TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
-- SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--
-- This script takes information about the current
-- version, compiler and compiler options and adds
-- them to a header filr
--

os.execute("svn info > premake_temp.txt")

f = io.open("premake_temp.txt", "r");

revision = -1
author = "unknown"
lastdate =  "unknown"
for line in f:lines() do
	local a, b, c = string.find(line, "Revision:%s*(%S+)")
	if a then
		revision = c
	end

	local a, b, c = string.find(line, "Author:")
	if a then
		author = line
	end

	local a, b, c = string.find(line, "Date:%s*(%S*)")
	if a then
		lastdate = "Last Changed Date:   " .. c
	end
end
f:close()

os.execute(arg[1] .. " --version > premake_temp.txt")
f = io.open("premake_temp.txt", "r");
compiler = f:read("*line")
f:close()

if compiler == nil then --try err stream, can't combine without knowing platform
	os.execute(arg[1] .. " --version 2> premake_temp.txt")
	f = io.open("premake_temp.txt", "r");
	compiler = f:read("*line")
	f:close()
end

f = io.open("info.h", "w")
f:write("const char* __info =\"\\\n")
f:write("MagLua Revision:     " .. revision .. [[\n\]] .. "\n")
f:write(author .. [[\n\]] .. "\n")
f:write("Compiler:            " .. compiler .. [[\n\]] .. "\n")
f:write([[";]] .. "\n")

f:write("const char* __rev  = \"" .. revision .. "\";\n")
f:write("const int   __revi =  " .. revision .. ";\n")

f:close()

-- This script helps to make MagLua-Win32 releases
-- build all and install in proper directories:
--     lua build.lua
-- build all/install cpu:
--     lua build.lua cpu
-- build exchange in cpu/install:
--     lua build.lua cpu exchange
--

configuration = "Release"
--configuration = "Debug"

category = arg[1] or ""
  module = arg[2] or ""

modules_common = {"encode","checkpoint","interpolate","random","timer"}
modules_cpu    = {"core", "longrange", "anisotropy", "appliedfield", "dipole", "disordereddipole", "exchange", "llg", "magnetostatics", "thermal"}
modules_cuda   = {"core_cuda", "longrange_cuda", "anisotropy_cuda", "appliedfield_cuda", "dipole_cuda", "exchange_cuda", "llg_cuda", "magnetostatics_cuda", "thermal_cuda"}

mod_path = {}
for k,v in pairs({"cpu", "cuda", "common"}) do
	os.execute("maglua --module_path " .. v .. " > temp.txt")
	local f = io.open("temp.txt", "r")
	mod_path[v] = f:read("*line")
	f:close()
end

for k,v in pairs(mod_path) do
	print(k,v)
end

function make_cmd(directory, module)
	return string.format([[cd modules\%s\%s && lua ..\..\..\make_vcxproj.lua %s]], directory, module, configuration)
end

-- os.execute = print

if category == "" or category == "common" then
	for k,v in pairs(modules_common) do
		if module == "" or module == v then
			os.execute(make_cmd("common", v))
			os.execute("copy ..\\Common\\"..v..".dll \"" .. mod_path["common"] .. "\"")
		end
	end
end

if category == "" or category == "cpu" then
	for k,v in pairs(modules_cpu) do
		if module == "" or module == v then
			os.execute(make_cmd("cpu", v))
			os.execute("copy ..\\Common\\"..v..".dll \"" .. mod_path["cpu"] .. "\"")
		end
	end
end

function strip_(line)
	return string.gsub(line, "_", "")
end

if category == "" or category == "cuda" then
	for k,v in pairs(modules_cuda) do
		if module == "" or module == v then
			os.execute(make_cmd("cuda", v))
			os.execute("copy ..\\Common\\"..strip_(v)..".dll \"" .. mod_path["cuda"] .. "\"")
		end
	end
end






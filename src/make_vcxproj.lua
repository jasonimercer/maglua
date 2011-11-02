-- 
-- This lua script processes a Makefile for Maglua and 
-- generates visual studio 10 project files
-- 

basedir=[[C:\programming\c\maglua\src]]
aid = basedir .. ";"
aid = aid .. basedir .. [[\..\Common;]]
aid = aid .. basedir .. [[\modules\common\encode;]]
aid = aid .. basedir .. [[\modules\cpu\core;]]


outputDirectory = basedir .. [[\..\Common\]]
additionalIncludeDirectories = aid
additionalLibraryDirectories = basedir .. [[\..\Common;]]
additionalDependencies = [[lua5.1.lib;libfftw3-3.lib;]]

m = io.open("Makefile", "r")

objects = {}
lines = {}
outs = {}
deps = {}
extra_includes = {}

function getList(line, suffix)
	suffix = suffix or ""
	local oo = {}
	while line do
		local a, b, c, d = string.find(line, "%s*(%S+)" .. suffix .. "(.*)")
		if c then
			table.insert(oo, c)
		end
		line = d
	end
	return oo
end

for line in m:lines() do
	local a, b, c = string.find(line, "OBJECTS%s*=%s*(.*)")
	if a then
		objects = getList(c, "%.o")
	end
	
	local a, b, c = string.find(line, "EXTRA_INCLUDE%s*=%s*(.*)")
	if a then
		extra_includes = getList(c)
	end
	
	local a, b, c = string.find(line, "LIBNAME%s*=%s*(.*)")
	if a then
		libname = c
	end
	
	local a, b, c = string.find(line, "BIN%s*=%s*(.*)")
	if a then
		binname = c
	end
	
	local a, b, c = string.find(line, "DEPEND%s*=%s*(.*)")
	if a then
		deps = getList(c)
	end
end

for k,v in pairs(extra_includes) do
	print(k,v)
	local a, b, c = string.find(v, "%s*%-I(.*)%s*")
	if a then
		local win32 = "$(ProjectDir)"  .. [[\]] .. string.gsub(c, "/", [[\]]) .. ";"
		print(win32)
		additionalIncludeDirectories = additionalIncludeDirectories .. win32
			
	end
end

for k,v in pairs(deps) do
	additionalDependencies = additionalDependencies .. v .. ".lib;"
end

if libname == nil and binname ~= nil then
	libname = binname
end

if libname == nil then
	error("Can't find name")
end

-- if string.lower(libname) ~= "encode" then
-- 	additionalDependencies = additionalDependencies .. "encode.lib;"
-- end

if binname ~= nil then
	configurationType = "Application"
	subsystem = "Console"
else
	configurationType = "DynamicLibrary"
	subsystem = "Windows"
end

header = [[<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Release|Win32">
      <Configuration>Release</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug|Win32">
      <Configuration>Debug</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <ItemGroup>]]

footer = [[  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{32C3BD91-FE3C-4F71-A64C-3812F37B58F9}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>LOWERCASENAME</RootNamespace>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'" Label="Configuration">
    <ConfigurationType>CONFIGURATIONTYPE</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'" Label="Configuration">
    <ConfigurationType>CONFIGURATIONTYPE</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings">
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <LinkIncremental>false</LinkIncremental>
    <OutDir>OUTPUTDIRECTORY</OutDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <LinkIncremental>true</LinkIncremental>
    <OutDir>OUTPUTDIRECTORY</OutDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;NDEBUG;_WINDOWS;_USRDLL;UPPERCASENAME_EXPORTS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      INCLUDEDIRECTORIES
	  ADDITIONALLIBRARYDIRECTORIES
    </ClCompile>
    <Link>
      <SubSystem>SUBSYSTEM</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      ADDITIONALDEPENDANCIES
      ADDITIONALLIBRARYDIRECTORIES
	  </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <ClCompile>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>WIN32;_DEBUG;_WINDOWS;_USRDLL;UPPERCASENAME_EXPORTS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      INCLUDEDIRECTORIES
	  ADDITIONALLIBRARYDIRECTORIES
    </ClCompile>
    <Link>
      <SubSystem>SUBSYSTEM</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      ADDITIONALDEPENDANCIES
	  ADDITIONALLIBRARYDIRECTORIES
      <ModuleDefinitionFile>
      </ModuleDefinitionFile>
      <DelayLoadDLLs>%(DelayLoadDLLs)</DelayLoadDLLs>
    </Link>
  </ItemDefinitionGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets">
  </ImportGroup>
</Project>
]]
  
  
lowercase = string.lower(libname)
uppercase = string.upper(libname)
includedir = [[<AdditionalIncludeDirectories>]] .. additionalIncludeDirectories .. [[%%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>]]
adddeps = [[<AdditionalDependencies>]] .. additionalDependencies .. [[%%(AdditionalDependencies)</AdditionalDependencies>]]
addlibdir = [[ <AdditionalLibraryDirectories>]] .. additionalLibraryDirectories .. [[%%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>]]

footer = string.gsub(footer, "UPPERCASENAME", uppercase)
footer = string.gsub(footer, "LOWERCASENAME", lowercase)
footer = string.gsub(footer, "INCLUDEDIRECTORIES", includedir)
footer = string.gsub(footer, "ADDITIONALDEPENDANCIES", adddeps)
footer = string.gsub(footer, "ADDITIONALLIBRARYDIRECTORIES", addlibdir)
footer = string.gsub(footer, "OUTPUTDIRECTORY", outputDirectory)
footer = string.gsub(footer, "CONFIGURATIONTYPE", configurationType)
footer = string.gsub(footer, "SUBSYSTEM", subsystem)

filename = lowercase .. ".vcxproj"

vcxproj = io.open(filename, "w")

vcxproj:write(header .. "\n")

for k,v in pairs(objects) do
	vcxproj:write([[    <ClInclude Include="]] .. v .. [[.h" />]] .. "\n")
end

vcxproj:write("  </ItemGroup>\n  <ItemGroup>\n")
for k,v in pairs(objects) do
	vcxproj:write([[    <ClCompile Include="]] .. v .. [[.cpp" />]] .. "\n")
end

vcxproj:write(footer .. "\n")

vcxproj:close()

table.insert(outs, filename)
print("Wrote `" .. filename .. "'")




filename = filename .. ".filters"

filters = io.open(filename, "w")


header = [[<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <Filter Include="Source Files">
      <UniqueIdentifier>{4FC737F1-C7A5-4376-A066-2A32D752A2FF}</UniqueIdentifier>
      <Extensions>cpp;c;cc;cxx;def;odl;idl;hpj;bat;asm;asmx</Extensions>
    </Filter>
    <Filter Include="Header Files">
      <UniqueIdentifier>{93995380-89BD-4b04-88EB-625FBE52EBFB}</UniqueIdentifier>
      <Extensions>h;hpp;hxx;hm;inl;inc;xsd</Extensions>
    </Filter>
    <Filter Include="Resource Files">
      <UniqueIdentifier>{67DA6AB6-F800-4c08-8B7A-83BB121AAD01}</UniqueIdentifier>
      <Extensions>rc;ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe;resx;tiff;tif;png;wav;mfcribbon-ms</Extensions>
    </Filter>
  </ItemGroup>
  <ItemGroup>]]
  
filters:write(header .. "\n")
for k,v in pairs(objects) do
	filters:write([[    <ClInclude Include="]] .. v .. [[.h" > <Filter>Header Files</Filter> </ClInclude>]] .. "\n")
end

filters:write([[  </ItemGroup>
  <ItemGroup>]] .. "\n")

for k,v in pairs(objects) do
	filters:write([[    <ClCompile Include="]] .. v .. [[.h" > <Filter>Source Files</Filter> </ClCompile>]] .. "\n")
end
  
filters:write([[  </ItemGroup>
</Project>]])

filters:close()

table.insert(outs, filename)
print("wrote `" .. filename .. "'")



filename = lowercase .. ".sln"
sln = io.open(filename, "w")

guid1 = "8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942"
guid2 = "32C3BD91-FE3C-4F71-A64C-3812F37B58F8"

sln:write([[Microsoft Visual Studio Solution File, Format Version 11.00
# Visual C++ Express 2010
Project("{]]..guid1..[[}") = "]]..lowercase..[[", "]]..lowercase..[[.vcxproj", "{]]..guid2..[[}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|Win32 = Debug|Win32
		Release|Win32 = Release|Win32
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{]]..guid2..[[}.Debug|Win32.ActiveCfg = Debug|Win32
		{]]..guid2..[[}.Debug|Win32.Build.0 = Debug|Win32
		{]]..guid2..[[}.Release|Win32.ActiveCfg = Release|Win32
		{]]..guid2..[[}.Release|Win32.Build.0 = Release|Win32
	EndGlobalSection
	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
EndGlobal
]])

sln:close()
table.insert(outs, filename)
print("wrote `"..filename.."'")

if os.getenv("HOME") ~= nil then
print("converting to DOS format: ")

for k,v in pairs(outs) do
	print("   > " .. v)
	local inp = assert(io.open(v, "rb"))
    local data = inp:read("*all")
	inp:close()
	
	data = string.gsub(data, "\n", "\r\n")
    
    local out = assert(io.open(v, "wb"))
	out:write(data)
    assert(out:close())
end
end

print("\nThis project can be compiled with the following command:")
print("msbuild /property:Configuration=Release")


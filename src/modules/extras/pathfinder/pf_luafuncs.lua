-- PathFinder
local MODNAME = "PathFinder"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time
local methods = {}


methods["findPath"] =
{
"Run a modified Bellman-Ford algorithm over sets of subdivided dodecahedrons to find a coarse minimum energy path between a pair of points.",
"2 Tables, 1 optional Integer: Each table contains sites, each site is 3 numbers representing a vector and a string naming the coordinate system. The integer can be between 1 and 5 inclusively (default 1) and determines the number of subdivisions of the dodecahedron representing a sphere. Larger values will take a considerably longer amount of time to solve. A single site system with a subdivision count of 5 will result in approximately 750 vertices, each with 7 neighbours. A 2 site system with a subdivision count of 5 will result in half a million vertices, each with 49 neighbours. ",
"1 Table: compatible with MEP:setInitialPath, represents the solved path from the start to the end.",
function(pf, a,b,n)
    local p = pf:_findBestPath(a,b,n)
    table.insert(p, 1, a)
    table.insert(p, b)
    return p
end
}




-- inject above into existing metatable for MEP operator
for k,v in pairs(methods) do
    t[k] = v[4]
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

		  -- fallback to old help if a case is not handled
		  if x == nil then -- Pathfinder overview
		      return [[Find paths on grids]], "", ""
		  end
		  return help(x)
	      end



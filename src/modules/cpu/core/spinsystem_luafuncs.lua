-- SpinSystem

local MODNAME = "SpinSystem"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

-- trying something new for style. Putting docs and code in a table which will later be 
-- used to build metatable and help system.
local methods = {} 
            

methods["netMoment"] = {
	"Calculate and return the net moment of a spin system",
	"N Optional Array.Double, M Optional Number: The optional double arrays scale each site by the product of their values, the optional numbers scales all sites by a single number. A combination of arrays and a single values can be supplied in any order.",
	"8 numbers: mean(x), mean(y), mean(z), vector length of {x,y,z}, mean(x^2), mean(y^2), mean(z^2), length of {x^2, y^2, z^2}",
	function(ss, ...)
		local nx, ny, nz = ss:nx(), ss:ny(), ss:nz()
		local wsReal = Array.DoubleWorkSpace(nx,ny,nz, "SpinSystem_Real_WS")

		local source_data =  {ss:spinArrayX(), ss:spinArrayY(), ss:spinArrayZ()}
		local res = {}

		for c=1,3 do
			source_data[c]:copy(wsReal)
			local scale = 1
			for i = 1, select('#',...) do
				local v = select(i,...)
				if type(v)==type(1.23) then
					scale = scale * v
				end
				if type(v)==type(wsReal) then
					wsReal:pairwiseMultiply(v, wsReal)
				end
			end

			wsReal:scale(scale)

			res[c] = {wsReal:sum(1), wsReal:sum(2)}
		end

		local x1,y1,z1 = res[1][1], res[2][1], res[3][1]
		local x2,y2,z2 = res[1][2], res[2][2], res[3][2]

		return x1, y1, z1, (x1^2 + y1^2 + z1^2)^(1/2), 
               x2, y2, z2, (x2^2 + y2^2 + z2^2)^(1/2)
		
	end
}

methods["difference"] =
{
    "Compute the sum of the magnitudes of the vector differences at each site between the calling and provided *SpinSystem*.",
    "1 SpinSystem: SpinSystem to compare",
    "1 Number or nil: Sum of magnitudes of vector differences if SpinSystems are the same size, nill otherwise.",
    function(ss, ssb)
	if not ss:sameDimensions(ssb) then
	    return
	end

	local x = ss:spinArrayX():pairwiseScaleAdd(-1, ssb:spinArrayX())
	local y = ss:spinArrayY():pairwiseScaleAdd(-1, ssb:spinArrayY())
	local z = ss:spinArrayZ():pairwiseScaleAdd(-1, ssb:spinArrayZ())

	x:pairwiseMultiply(x, x)
	y:pairwiseMultiply(y, y)
	z:pairwiseMultiply(z, z)

	x:pairwiseScaleAdd(1, y, x)
	x:pairwiseScaleAdd(1, z, x)

	x:elementsRaisedTo(0.5, x)

	return x:sum()

    end
}

methods["sameDimensions"] =
{
    "Test if the calling SpinSystem is of the same dimensions as the provided SpinSystem",
    "1 SpinSystem: SpinSystem to compare.",
    "1 Boolean: True is dimensions match, False otherwise",
    function(ss, ssb)
	if ss:nx() ~= ssb:nx() then
	    return false
	end
	if ss:ny() ~= ssb:ny() then
	    return false
	end
	if ss:nz() ~= ssb:nz() then
	    return false
	end
	return true
    end
}


-- inject above into existing metatable for SpinSystem
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
        if x == nil then
			return help()
        end
        return help(x)
end

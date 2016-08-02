-- SpinSystem

local MODNAME = "SpinSystem"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time

-- trying something new for style. Putting docs and code in a table which will later be 
-- used to build metatable and help system.
local methods = {} 

local type_text = type("")
local type_number = type(1)
local type_table = type({})
local type_function = type(type)
local type_nil = type(nil)            

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
                                local type_v = type(v)
				if type_v==type_number then
					scale = scale * v
				end
				if type_v==type(wsReal) then
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

methods["positionToIndex"] =
{
    "Convert a position to an index between 1 and xyz",
    "1 3Vector: Position to convert. If any dimension is out of range, periodic coordinates will be applied.",
    "1 Integer: Index corresponding to position.",
    function(ss, x,y,z)
        if type(x) == type_table then
            return ss:positionToIndex(x[1] or 1, x[2] or 1, x[3] or 1)
        end

        x=x or 1
        y=y or 1
        z=z or 1

        while x < 1 do x = x + ss:nx() end
        while y < 1 do y = y + ss:ny() end
        while z < 1 do z = z + ss:nz() end

        while x > ss:nx() do x = x - ss:nx() end
        while y > ss:ny() do y = y - ss:ny() end
        while z > ss:nz() do z = z - ss:nz() end

        return (z-1) * (ss:nx() * ss:ny()) + (y-1) * (ss:nx()) + x
    end
}

-- getName now defined in C
local getName = _getName
_getName = nil


methods["ensureSlotExists"] =
{
"Ensure the SpinSystem has a field slot with the given name.",
"1 String or an object with a slot name: The field name.",
"",
function(ss, n)
    return ss:_ensureSlotExists( getName(n) )
end
}


methods["setSlotUsed"] =
{
"Set an internal variable. If true this field type will be added in the sum fields method.",
"1 String or an object with a slot name, 0 or 1 Boolean: A field name and a flag to include or exclude the field in the summation method. Default value is true.",
"",
function(ss, n, b)
    if type(b) == type_nil then
	return ss:_setSlotUsed( getName(n) )
    else
	return ss:_setSlotUsed( getName(n), b )
    end
end
}
methods["slotUsed"] =
{
"Determine if an internal field slot has been set.",
"1 String or an object with a slot name: A field name.",
"1 Boolean: The return value.",
function(ss,n)
    return ss:_slotUsed( getName(n) )
end
}



methods["setFieldArrayX"] =
{
"Set the X components of the field vectors for a given type.",
"1 String or an object with a slot name, 1 Array: The name of the field, the new X components of the field for each sites.",
"",
function(ss, n, v)
    return ss:_setFieldArrayX( getName(n), v )
end
}
methods["setFieldArrayY"] =
{
"Set the Y components of the field vectors for a given type.",
"1 String or an object with a slot name, 1 Array: The name of the field, the new Y components of the field for each sites.",
"",
function(ss, n, v)
    return ss:_setFieldArrayY( getName(n), v )
end
}
methods["setFieldArrayZ"] =
{
"Set the Z components of the field vectors for a given type.",
"1 String or an object with a slot name, 1 Array: The name of the field, the new X components of the field for each sites.",
"",
function(ss, n, v)
    return ss:_setFieldArrayZ( getName(n), v )
end
}



methods["field"] =
{
"Get the field at a site due to an interaction",
"1 String or an object with a slot name, 1 3Vector: The first argument identifies the field interaction type, must match a :slotName() of an applied operator. The second argument selects the lattice site.",
"4 Numbers: The field vector at the site and the magnitude fo the field",
function(ss,n,...)
    local type_n = type(n)
    if type_n == type_number or type_n == type_table then
	return ss:field("Total", n, ...)
    end
    return ss:_field( getName(n), ...)
end
}



methods["fieldArrayX"] =
{
"Get the X components of the field vectors for a given type.",
"1 String or an object with a slot name: The name of the field.",
"1 Array: The X components of the field for each sites.",
function(ss,n)
    return ss:_fieldArrayX( getName(n) )
end
}
methods["fieldArrayY"] =
{
"Get the Y components of the field vectors for a given type.",
"1 String or an object with a slot name: The name of the field.",
"1 Array: The Y components of the field for each sites.",
function(ss,n)
    return ss:_fieldArrayY( getName(n) )
end
}
methods["fieldArrayZ"] =
{
"Get the Z components of the field vectors for a given type.",
"1 String or an object with a slot name: The name of the field.",
"1 Array: The Z components of the field for each sites.",
function(ss,n)
    return ss:_fieldArrayZ( getName(n) )
end
}




methods["zeroField"] =
{
"Zero all values for a given field type",
"1 String or an object with a slot name: The name of the field.",
"",
function(ss,n)
    local gn = getName(n)
    ss:_fieldArrayX(gn):zero()
    ss:_fieldArrayY(gn):zero()
    ss:_fieldArrayZ(gn):zero()
end
}



local function zero(ss,n)
    local gn = getName(n)
    ss:_ensureSlotExists(gn)
    ss:_setSlotUsed(gn, false)
    --ss:zeroField(n)
    ss:_fieldArrayX(gn):zero()
    ss:_fieldArrayY(gn):zero()
    ss:_fieldArrayZ(gn):zero()
    -- print("Zero'd " .. n)
end

methods["resetFields"] =
{
"Zero some or all of the fields and exclude them from future sums until they are added to the sum list either via applying new fields or setting the internal flag explicitly with :setSlotUsed.",
"Nothing or a table of Strings or an objects with a slot name: Fields to reset (default all registered slots).",
"",
function(ss,t)
    if t == nil then
        ss:_resetFields()
	--for k,v in pairs(ss:slots()) do
	--    zero(ss,v)
	--end
	return
    end

    if type(t) == type_table then
	for k,v in pairs(t) do
	    zero(ss,v)
	end
	return
    end

    error("Don't know how to deal with input to resetFields. Expected nothing or a table")
end
}

local function sumFieldsCustom(ss,dest_name, src_names)
    ss:_ensureSlotExists(dest_name)
    
    local dest_x = ss:_fieldArrayX(dest_name)
    local dest_y = ss:_fieldArrayY(dest_name)
    local dest_z = ss:_fieldArrayZ(dest_name)

    dest_x:zero()
    dest_y:zero()
    dest_z:zero()

    for k,src_name in pairs(src_names) do
        local factor = 1
        if type(src_name) == type_table then
            factor = src_name[1]
            src_name = src_name[2]
        end

	if ss:_slotUsed(src_name) then
	    dest_x:pairwiseScaleAdd(factor, ss:_fieldArrayX(src_name), dest_x)
	    dest_y:pairwiseScaleAdd(factor, ss:_fieldArrayY(src_name), dest_y)
	    dest_z:pairwiseScaleAdd(factor, ss:_fieldArrayZ(src_name), dest_z)
	end
    end

end

methods["sumFields"] =
{
"Sum all fields or all given fields into a given slot or the `Total' slot",
"1 or 0 Strings or an object with a slot name, 1 or 0 Tables of Strings or objects with a slot name or pairs of numbers and names or objects: Destination slot name for summation (default `Total'), source terms to be included in the summation (default all terms but the `Total' term). If a table of pairs is provided then the number will scale the field in the sum.",
"",
function(ss, a,b)
    if a == nil then
        return ss:_sumFields() -- vanilla sumFields
    end

    local type_a = type(a)
    if type_a == type_table then
        local src_names = {}
        for k,v in pairs(a) do
            src_names[k] = getName(v)
        end
        sumFieldsCustom(ss, "Total", src_names)
    else
        local dest_name = getName(a)
        local src_names

        if type(b) == type_table then
            src_names = {}
            for k,v in pairs(b) do
                if type(v) == type_table then
                    src_names[k] = {v[1], getName(v[2])}
                else
                    src_names[k] = getName(v)
                end
            end
        else
            src_names = ss:slotsNotTotal()
        end

        sumFieldsCustom(ss, dest_name, src_names)
    end

end
}












-- inject above into existing metatable for SpinSystem
for k,v in pairs(methods) do
    t[k] = v[4]
end

-- backup old help function for fallback
local help = MODTAB.help

-- create new help function for methods above
MODTAB.help = 
function(x)
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

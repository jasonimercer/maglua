-- This file provides a function `MakeMacro' that converts a *SpinSystem* and operators into data and operators where sites may be combined into macro-sites. 
-- The function arguments are flexible but the first must be a mapping table. The following are data and operators. One must be a SpinSystem, the others can be Exchange, Anisotropy, Applied and Thermal. 
-- The return values will be a function that does a reverse mapping lookup and the transformed input in the order that they were supplied. 
-- The mapping table is a list of macro sites where a macro site is a list of sites to be combined into a site. These sites can be in 1, 2 or 3 dimensional format. Consider the example below:
--
-- <pre>
-- mapping = { {{1}}, {{2}}, {{3}, {4}}, {5} }
-- </pre>
-- This 1D mapping table will map the old site 1 onto the new site 1, the old site 2 onto the new site 2, the old sites 3 and 4 onto the new site 2 and the old site 5 onto the new site 4.
-- Old sites mapped to multiple new sites is unsupported. Invalid old sites are unsupported.
--
-- The first argument that is returned, the reverseMappingLookup function takes a single old site as a table and returns 3 integers. The first is the index of the new macro-site that contains the old site, the second is the index inside the new macro-site where the old site is contained and the last is the number of sites in the new macro-site.
-- 
-- Example use:
-- <pre>
-- invmap, ssMacro, exMacro, aniMacro = MakeMacro(mapping, ss, ex, ani)
-- </pre>
-- 
-- If sites in the original system contain extraData then the first data in the mapping will be assigned to the extraData of the new system.


function MakeMacro(_mapping, arg2, arg3, arg4, arg5, arg6)
	if type(_mapping) ~= "table" then
		error("Mapping must be a table")
	end
	
	
	local fargs = {nil, arg2, arg3, arg4, arg5, arg6}
	local ss, ex, ani, thermal, zee
	local ss_idx, ex_idx, ani_idx, thermal_idx, zee_idx
	local mapping = _mapping or {}
	
	local exMacro, ssMacro
	local newMoments = {}
	
	for i=2,6 do
		if fargs[i] then
			local mt = getmetatable(fargs[i])
			if mt == SpinSystem.metatable() then
				ss = fargs[i]
				ss_idx = i
			end
			if mt == AppliedField.metatable() then
				zee = fargs[i]
				zee_idx = i
			end
			if mt == Exchange.metatable() then
				ex = fargs[i]
				ex_idx = i
			end
			if mt == Anisotropy.metatable() then
				ani = fargs[i]
				ani_idx = i
			end
			if mt == Thermal.metatable() then
				thermal = fargs[i]
				thermal_idx = i
			end
		end
	end
	
	local function reverseMappingLookup(old_site)
		local ox = old_site[1] or 1
		local oy = old_site[2] or 1
		local oz = old_site[3] or 1
		for i=1,table.maxn(mapping) do
			for j=1,table.maxn(mapping[i]) do
				local nx = mapping[i][j][1] or 1
				local ny = mapping[i][j][2] or 1
				local nz = mapping[i][j][3] or 1
				
				if nx==ox and ny==oy and nz==oz then
					return i, j, table.maxn(mapping[i])
				end
			end
		end
-- 		print("failed to find ", table.concat(old_site, ","))
	end

	-- if ss exists then we will make new ssMacro
	if ss ~= nil then
		for i=1,table.maxn(mapping) do
			local m = 0
			for j=1,table.maxn(mapping[i]) do
				local _,_,_,mm = ss:spin(mapping[i][j])
				m = m + mm
			end
			newMoments[i] = m
		end
		
		
		local macroSites = table.maxn(mapping)
		ssMacro = SpinSystem.new(macroSites)

		-- setting spin direction and magnitudes in the macro-model
		for i=1,macroSites do
			local mx, my, mz, mm = 0,0,0,0
			local extra_data = nil
			for j=1,table.maxn(mapping[i]) do
				local x,y,z,m = ss:spin(mapping[i][j])
				mx,my,mz,mm = mx+x, my+y, mz+z, mm+m
				extra_data = extra_data or ss:extraData(mapping[i][j])
			end

			ssMacro:setSpin({i}, {mx,my,mz}, mm) --make 1:1 mappings and macro-spins
			ssMacro:setExtraData({i}, extra_data)
		end
		
		ssMacro:setTimeStep(ss:timeStep())
		ssMacro:setAlpha(ss:alpha())
		ssMacro:setGamma(ss:gamma())
		ssMacro:setTime(ss:time())
	else
		error("Soruce spin system not found in arguments")
	end
	
	-- if exchange op and ss exists, make new one based on mapping
	if ex ~= nil and ss ~= nil then
		exMacro = Exchange.new(ssMacro)
		-- we will iterate through the original pathways 
		-- and make adjustments for the new mapping.
		for i=1,ex:numberOfPaths() do
			local src, dst, strength = ex:path(i)
			local src_new_idx, src_new_part, src_new_size = reverseMappingLookup(src)
			local dst_new_idx, dst_new_part, dst_new_size = reverseMappingLookup(dst)
			
			-- exclude intra-macro interaction
			if src_new_idx ~= nil and dst_new_idx ~= nil then
				local _,_,_,mm = ss:spin(src)
				local _,_,_,dd = ss:spin(dst)
				-- scaling strength by the fraction that the 
				-- source makes up vs the macrospin
				strength = strength * ( (mm * dd) / (newMoments[src_new_idx] * newMoments[dst_new_idx]) )
				if src_new_idx ~= dst_new_idx then
					exMacro:add({src_new_idx}, {dst_new_idx}, strength)
				end
			end
		end
		-- combine all repeats (as in borders between macrospins)
-- 		print("ex delta", exMacro:mergePaths())
	end
	
	if ani ~= nil and ss ~= nil then
		aniMacro = Anisotropy.new(ssMacro)
		
		for i=1,ani:numberOfAxes() do
			local site, direction, strength = ani:axis(i)
			local new_idx, new_part, new_size = reverseMappingLookup(site)
			if new_idx then
-- 				print(new_idx, new_part, new_size)
				aniMacro:add({new_idx}, direction, strength/new_size)
			end
		end
		aniMacro:mergeAxes()
	end
	
	if thermal ~= nil and ss ~= nil then
		thermalMacro = Thermal.new(ssMacro, thermal:random())
		thermalMacro:setTempterature(thermal:temperature())
	end
		
	if zee ~= nil and ss ~= nil then
		zeeMacro = AppliedField.new(ssMacro)
	end
	
	
	-- The following builds the return statement so that it matches the 
	-- input to this function.
	local return_values = {reverseMappingLookup}
	
	if ss_idx      then return_values[ss_idx] = ssMacro	end
	if ex_idx      then return_values[ex_idx] = exMacro	end
	if ani_idx     then return_values[ani_idx] = aniMacro	end
	if thermal_idx then return_values[thermal_idx] = thermalMacro	end
	if zee_idx     then return_values[zee_idx] = zeeMacro	end
	
	
	-- flatten out a list into return values
	local function return_list(t, n)
		if n == 0 then
			return nil
		end
		local x = t[1]
		for i=1,10 do
			t[i] = t[i+1]
		end
		return x, return_list(t, n-1)
	end
	
	return return_list(return_values, 10)
end

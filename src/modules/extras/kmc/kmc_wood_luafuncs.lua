-- This file provides a KMC.Wood implementation to the KMC framework. The
-- internal event and startup functions are provided


-- here are the fallback functions we will extend
parent_class_help = KMC.help
parent_class_new = KMC.new

KMC.Wood = {}
KMC.Wood.metatable = KMC.metatable

KMC.Wood.help = function(x)
	if x == nil then
		return 
			"Wood's solution implementation for the KMC energy barrier. See ref #reference for details. TODO: reference",
		    "1 *SpinSystem*: "
	end
	return parent_class_help(x)
end

-- this is where we will add in the wood functionality
KMC.Wood.new = function(ss)
	if ss == nil then
		error("KMC.Wood.new requires a SpinSystem as a constructor argument")
	end


	-- creating startup function
	-- we want to make the wood object and alter the 
	-- demag tensor once so we'll test to see if we have the 
	-- wood's object created
	function startupFunction(kmc, ss)
		local user_data = (kmc:customData() or {})
		if user_data.wood == nil then
			user_data.wood = Wood.new(ss)
			
			for k,demag_op in pairs(kmc:fields("Magnetostatics2D")) do
				user_data.wood:adjustMagnetostatics(demag_op)
			end
			kmc:setCustomData(user_data)
		end

		ss:resetFields()
		-- apply all the field (except ani and thermal)
		for k,field_op in pairs(kmc:fields()) do
			local sn = field_op:slotName()
			if sn ~= "Anisotropy" and sn ~= "Thermal" then
				field_op:apply(ss)
			end
		end
		ss:sumFields()

		local ani_table = kmc:fields("Anisotropy")
		if ani_table[1] == nil then
			error("Wood's solution needs an anisotropy operator in the fields")
		end
		local ani = ani_table[1]
		
		local mag_table = kmc:fields("Magnetostatics2D")
		if mag_table[1] == nil then
			error("Wood's solution needs a Magnetostatics2D operator in the fields")
		end
		local mag = mag_table[1]

		user_data.wood:calculateEnergyBarriers(ss, ani, mag)
	end
	
	
	-- Tim's f0 code
	-- calc f0
	-- the quantities below are in units involving inverse density
	function f0_calc(ss,hz,h,k,m,kT,D,  mag)
        local gx,gy,gz, grainSize = mag:grainSize(1)
        k = k/grainSize
        m = m/grainSize
        D = D*grainSize
        local Vol = grainSize
        local alpha = ss:alpha()
        local gamma = ss:gamma()
        local Hk = 2*k/m + D*m

        local f0 = 2*alpha*gamma*Hk * math.sqrt( 4*k*Vol/math.pi/kT )

        if hz <  0 then f0 = f0*(1-h/Hk)^2 end
        if hz >= 0 then f0 = f0*(1+h/Hk)^2 end

        return f0
	end


	-- creating event function
	-- this function will return the energy barrier and attempt frequency for 
	-- an event as well as code that will execute the event
	function eventFunction(kmc, ss, pos)
		local user_data = kmc:customData()
		local wood = user_data.wood

		local i,j,k = pos[1], pos[2], pos[3]
		local hx,hy,hz, hh   = ss:field("Total", pos)
		local kx,ky,kz, kk   = ani:get(pos)
		local sx,sy,sz, smag = ss:spin(pos)

		local mag_ops = kmc:fields("Magnetostatics2D") 
		local mag = mag_ops[1]
		local gx,gy,gz, grainSize = mag:grainSize(k)
		local EB = wood:energyBarrier(i,j,k)
		local f0 = f0_calc(ss,hz, hh, kk, smag, kmc:temperature(), wood:getDemag(k)/grainSize, mag)


		-- returning a function with pos and orientation enclosed that will act on a spinsystem
		local function event(ss)
			ss:setSpin(pos, {sx,sy,-sz})
		end

		-- as required, time and event
		local kT = kmc:temperature()
		local rng = kmc:rng()
		local decay = f0 * math.exp(-EB/kT)
		local r = rng:rand()

		local time = -math.log(r) / decay


		return time, event

	end


	kmc = parent_class_new()

	-- now we will add the above startup and event functions to the kmc object
	-- this provides the core to the framework and makes this object a usable KMC object
	kmc:setStartupFunction(startupFunction)
	kmc:setEventFunction(eventFunction)

	return kmc
end



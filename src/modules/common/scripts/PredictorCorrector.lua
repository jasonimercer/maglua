-- This support script implements predictor-corrector integration.
-- The function returns a step function and has the form:
-- <pre>function make_pc_step_function(ss, calcField, llg, tol, optional_temp)</pre>
-- where ss is a *SpinSystem*, calcField is a function that calculates the determinstic portion of the effective field, llg is a LLG operator and the optional parameter optional_temp is a temperature operator. 
--
-- Example:
-- <pre>
-- dofile("maglua://PredictorCorrector.lua")
-- 
-- ss   = SpinSystem.new(40,40)
-- ex   = Exchange.new(ss)
-- dip  = Dipole.new(ss)
-- ani  = Anisotropy.new(ss)
-- temp = Thermal.new(ss, Random.Isaac.new())
-- llg  = LLG.Quaternion.new()
--
-- --1D interpolation object used as {time, temperature}
-- temperature = Interpolate.new({{0,20}, {20,2}, {198,0}, {1e8,0}})
--
-- max_time = 200
-- ss:setTimeStep(5e-2)
-- ex_str = 1/2
-- ani_str = 5/2
--
-- for i=1,ss:nx() do
-- 	for j=1,ss:ny() do
-- 		ss:setSpin({i,j}, {1,0,0})
-- 
-- 		ex:add({i,j}, {i+1,j}, ex_str)
-- 		ex:add({i,j}, {i-1,j}, ex_str)
-- 		ex:add({i,j}, {i,j+1}, ex_str)
-- 		ex:add({i,j}, {i,j-1}, ex_str)
-- 
-- 		ani:add({i,j}, {0,0,1}, ani_str)
-- 	end
-- end
-- 
-- function calcField(ss)
-- 	ss:resetFields()
-- 	ex:apply(ss)
-- 	dip:apply(ss)
-- 	ani:apply(ss)
-- 	ss:sumFields()
-- end
-- 
-- step = make_pc_step_function(ss, calcField, llg, 0.001*40^2, temp)
-- 
-- while ss:time() < max_time do
-- 	temp:set(temperature:value(ss:time()))
-- 	step()
-- end
-- </pre>

function make_pc_step_function(s_, calcField_, llg_, tol_, optional_temp_)
	local s = s_
	local ssPredict = s:copy()
	local ssCorrect = s:copy()
	local llg = llg_
	local calcFields = calcField_
	local optional_temp = optional_temp_
	local tol = tol_


	local function step(ss, skip_temperature)
		local last_same = -1
		local function same(ssA, ssB)
			local dx, dy, dz, dxyz = ssA:diff(ssB)
			if dxyz == last_same then --not improving
				error("Failing to converge in Predictor Corrector, timestep too large?", 6)
			end
			last_same = dxyz
			return dxyz < tol
		end
		ss = ss or s
		-- ssPredict = ss + dt ss'
		calcFields(ss)
		llg:apply(ss, ss, ssPredict)

		repeat
			-- ssCorrect = ss + dt 1/2(ss' + ssPredict')
			llg:apply(ss,  1/2, ss,  ssCorrect)
			calcFields(ssPredict)
			llg:apply(ssCorrect, 1/2, ssPredict, ssCorrect)

			-- swap them: use corrector for next prediction
			ssPredict, ssCorrect = ssCorrect, ssPredict 
		until same(ssPredict, ssCorrect)

		ssPredict:copySpinsTo(ss)
		ss:setTime(ssPredict:time())

		if optional_temp and not skip_temperature then
			ss:resetFields()
			optional_temp:apply(ss)
			ss:sumFields()
			llg:apply(ss, false) --false = don't advance timestep
		end
	end

	return step
end

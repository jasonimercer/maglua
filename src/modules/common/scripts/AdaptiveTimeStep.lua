-- This script takes a step function and wraps it in an adaptive timestep scheme halfing or doubling the timestep as required to acheive a desired tolerance.
-- Function form:
-- <pre>function make_adapt_step_function(ss, step, tolerance, dynamics, llg, optional_thermal_operator)</pre>
-- where ss is a *SpinSystem*, step is a function that integrates over a timestep, tolerance is used to determine if two systems are the same, dynamics is an optional function (can be nil) that takes a *SpinSystem* and modifies the environment, llg is an LLG operator (only required if a thermal operator is specified) and optional_thermal_operator is a thermal operator.


--function make_adapt_step_function(ss_, step_, tolerance, dynamics_, llg_, optional_thermal_operator)
function make_adapt_step_function(a1, a2, a3, a4, a5, a6)
	-- working on making args adaptive
	local args = {}
	args["userdata"] = {}
	args["function"] = {}
	args["number"] = {}

	local a = {a1,a2,a3,a4,a5,a6}

	for k,v in pairs(a) do
		local t = type(v)
		args[t] = args[t] or {}
		table.insert(args[t], v)
	end

	local ss_ = args["userdata"][1]
	local step_ = args["function"][1]
	local tolerance = args["number"][1]
	local dynamics_ = args["function"][2]
	local llg_ = args["userdata"][2]
	local optional_thermal_operator = args["userdata"][3]

	local tol = tolerance

	local ss  = ss_
	local ss2 = ss:copy()
	local ss3 = ss:copy()
	local llg = llg_
	local dynamics = dynamics_ or function() end
	
	local optional_temp = optional_thermal_operator
	
	local step = step_
	
	local function same(ssA, ssB)
		local dx, dy, dz, dxyz = ssA:diff(ssB)
		return dxyz < tol, (tol-dxyz)/tol
	end
	
	local function adapt_step(s, skip_temperature)
		s = s or ss
		
		ss2:setTime(s:time())
		ss3:setTime(s:time())

		ss2:setTimeStep(s:timeStep())
		ss3:setTimeStep(s:timeStep()/2)

		s:copySpinsTo(ss2)
		s:copySpinsTo(ss3)
		
		
		dynamics(ss2)
		step(ss2, true) -- ss2 = full step, true = no thermal contrib
	

		dynamics(ss3)  -- 2 half steps
		step(ss3, true)
		dynamics(ss3)
		step(ss3, true)
		
		local are_same, ee = same(ss2,ss3)
                print(are_same, ee, s:timeStep())
		if are_same then
			ss3:copySpinsTo(s)
			if optional_temp and not skip_temperature then
				dynamics(s)
				s:resetFields()
				optional_temp:apply(s)
				s:sumFields()
				llg:apply(s, false) --false = don't advance timestep
			end
			s:setTime(ss3:time())
			
-- 			print("good:", ee)
			ee = ee^2
			if ee > 0.5 then ee = 0.5 end
                        if ee < 0.1 then ee = 0.1 end
			local adjust = 1+ee  - 0.1
			s:setTimeStep(s:timeStep() * adjust) --stride longer next time
-- 			print("good", adjust)
		else
 			print("","","","","bad:", ee)
-- 			error("BAD!!")
			s:setTimeStep(s:timeStep() * 0.5) --stride shorter next time
		end
		
		return are_same
-- 		print("TS", s:timeStep())
	end
	return adapt_step
end












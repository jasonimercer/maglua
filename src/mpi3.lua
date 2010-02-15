name = mpi.get_processor_name()
rank = mpi.get_rank()
size = mpi.get_size()

nmacro     = 10
Jmacro     = 0.1

nmicro     = 10
Jmicro     = 1.0

dtmicro    = 0.05
alphamicro = 0.5
Mmicro     = 1
runmicro   = 0.1 --run micro for this long
runmacro   = 0.3 --run macro for this long

exch_dir = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}

-- setup macro spin system at rank(1)
if rank == 1 then
	ss_macro      = SpinSystem.new(nmacro, nmacro, nmacro)
	ex_macro      =   Exchange.new(nmacro, nmacro, nmacro)

	for i=1,nmacro do
		for j=1,nmacro do
			for k=1,nmacro do
				-- pbc Jmacro
				for _,d in pairs(exch_dir) do
					dx, dy, dz = i+d[1], j+d[2], k+d[3]
					ex_macro:addPath(i, j, k, dx, dy, dz, Jmicro)
				end
			end
		end
	end
end

my_ss     = {} --the micro spin systems will go here, flat list
my_ss_xyz = {} --the micro spin systems will go here, indexed by location
owner_map = {} --lists who owns which micro system

owner = 1
n = nmicro
for I=1,nmacro do
	owner_map[I] = {}
	my_ss_xyz[I] = {}
	for J=1,nmacro do
		owner_map[I][J] = {}
		my_ss_xyz[I][J] = {}
		for K=1,nmacro do
			owner_map[I][J][K] = owner
			if owner == rank then
				pos      = {I, J, K}
				spinsys  =   SpinSystem.new(n,n,n)
				exchange =     Exchange.new(n,n,n)
				thermal  =      Thermal.new(n,n,n)
				applied  = AppliedField.new(n,n,n)
				llg      =          LLG.new("Cartesian")

				llg:setAlpha(alphamicro)
				llg:setTimeStep(dtmicro)
				llg:setGamma(2.0)

				ss = {owner, pos, spinsys, exchange, thermal, applied, llg}
				table.insert(my_ss, ss)
				my_ss_xyz[I][J][K] = ss
			end

			owner = owner + 1
			if owner == size+1 then
				owner = 1
			end
		end
	end
end

--setup each micro spin system
for a,b in ipairs(my_ss) do
	s = b[3]
	e = b[4]

	for i=1,n do
		for j=1,n do
			for k=1,n do
				s:setSpin(i, j, k, 0, 0, Mmicro)

				-- non-pbc Jmicro
				for _,d in pairs(exch_dir) do
					dx, dy, dz = i+d[1], j+d[2], k+d[3]
					if e:member(dx, dy, dz) then
						e:addPath(i, j, k, dx, dy, dz, Jmicro)
					end
				end
			end
		end
	end
end



-- -- maglua supports 3 types of random number generators
-- -- Isaac            - http://burtleburtle.net/bob/rand/isaacafa.html
-- -- MersenneTwister  - http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
-- -- CRandom          - basic C random number generator
rng     = Random.new("Isaac")

 
-- f = io.open(filename, "w")
-- f:write("# this data is from the script " .. argv[2] .. "\n")
-- f:write("# Working on Tc vs J\n")
-- f:write("# time  temp  <Mx/Ms>  <My/Ms>  <Mz/Ms>  <(Mz/Ms)^2>-<(Mz/Ms)>^2\n")
-- f:flush()
-- 

-- a function that steps a spin system 1 dt
function step(t)
	--{owner, pos, spinsys, exchange, thermal, applied, llg}
	t[5]:setTemperature(0.4)

	t[3]:zeroFields()           -- ss
	t[4]:apply(t[3])            -- exchange - ss
	t[5]:apply(t[7], rng, t[3]) -- thermal  - llg, rng, ss
	t[3]:sumFields()

	t[7]:apply(t[3])
end

function time()
	if my_ss[1] then
		return my_ss[1][3]:time()
	else
		return nil
	end
end

while time() < runmacro do
	now = time()
	-- run all the little system "runmicro" time
	while now + runmicro > time() do
		for i,t in ipairs(my_ss) do
			step(t)
		end

		if rank == 1 then
			print(time())
		end
	end

	-- collect grain magnetizations
	-- clients send data to rank(1) process
	if rank == 1 then
		for i,t in ipairs(my_ss) do
			pos = t[2]
			px, py, pz = pos[1], pos[2], pos[3]
			mx, my, mz = t[3]:netMag()

			ss_macro:setSpin(px, py, pz, mx, my, mz)
		end
			
		for c=2,size do
			repeat
				px, py, pz, mx, my, mz = mpi.recv(c)
				if px then
					ss_macro:setSpin(px, py, pz, mx, my, mz)
				end
			until px == nil 
		end

		--now have all the graininfo. Make an exchange field calculation
		ss_macro:zeroFields()
		ex_macro:apply(ss_macro)
		ss_macro:sumFields()

		--{owner, pos, spinsys, exchange, thermal, applied, llg}
		-- and send fields to clients
		for i=1,nmacro do
			for j=1,nmacro do
				for k=1,nmacro do
					owner = owner_map[i][j][k]
					hx, hy, hz = ss_macro:getField("Total", i, j, k)
					if owner == 1 then
						my_ss_xyz[i][j][k][6]:set(hx, hy, hz) --
					else
						mpi.send(owner, i, j, k, hx, hy, hz)
					end
				end
			end
		end

		for i=2,size do
			mpi.send(i, nil, nil, nil, nil, nil, nil) -- no more data
		end
	else
		--send grain info
		for i,t in ipairs(my_ss) do
			pos = t[2]
			px, py, pz = pos[1], pos[2], pos[3]
			mx, my, mz = t[3]:netMag()
			
			print("A", rank, "presend")
			mpi.send(1, px, py, pz, mx, my, mz)
			print("A", rank, "postsend")
		end
		print("A", rank, "presend")
		mpi.send(1, nil) --end of data
		print("A", rank, "postsend")

		--get fields back
		repeat
			i, j, k, hx, hy, hz = mpi.recv(1)

			if i then
				my_ss_xyz[i][j][k][6]:set(hx, hy, hz)
			end
		until i == nil --end of data
	end
end

print(name, rank, "done")

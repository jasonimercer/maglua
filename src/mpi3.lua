rng  = Random.new("Isaac")
name = mpi.get_processor_name()
rank = mpi.get_rank()
size = mpi.get_size()

nmacro     = 10
Jmacro     = 10

nmicro     = 10
Jmicro     = 1.0

tempmicro  = 2.0
dtmicro    = 0.00001
alphamicro = 1.0
Mmicro     = 1

runmicro   = 0.0002 --run micro for this long
runmacro   = 0.0050 --run macro for this long

filename_prefix = "mpitest"

micro_count= {} --how many micro systems per process


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
			if 	micro_count[owner] then
				micro_count[owner] = micro_count[owner] + 1
			else
				micro_count[owner] = 1
			end

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

function random_spin(M)
	local mx = rng:normal()
	local my = rng:normal()
	local mz = rng:normal()
	local mm = math.sqrt(mx^2 + my^2 + mz^2)
	return M*mx/mm, M*my/mm, M*mz/mm
end

--setup each micro spin system
for a,b in ipairs(my_ss) do
	s = b[3]
	e = b[4]

	for i=1,n do
		for j=1,n do
			for k=1,n do
				local mx, my, mz = random_spin(Mmicro)
				s:setSpin(i, j, k, mx, my, mz)

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


 
-- a function that steps a spin system 1 dt
function step(t)
	local spinsys = t[3]
	local exchange= t[4]
	local thermal = t[5]
	local applied = t[6]
	local llg     = t[7]

	thermal:setTemperature(tempmicro)

	spinsys:zeroFields()
	exchange:apply(spinsys)
	applied:apply(spinsys)
	thermal:apply(llg, rng, spinsys)
	spinsys:sumFields()

	llg:apply(spinsys)

-- 	print(spinsys:spin(1,1,1))
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

	for t=1,runmicro/dtmicro do
		for i,t in ipairs(my_ss) do
			step(t)
		end
	end

	--mpi.barrier()

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
			for i = 1,micro_count[c] do
				px, py, pz, mx, my, mz = mpi.recv(c)
				ss_macro:setSpin(px, py, pz, mx, my, mz)
			end
		end
	else
		--send grain net magnetization
		for i,t in ipairs(my_ss) do
			pos = t[2]
			px, py, pz = pos[1], pos[2], pos[3]
			mx, my, mz = t[3]:netMag()
			mpi.send(1, px, py, pz, mx, my, mz)
		end
	end

	--mpi.barrier()

	if rank == 1 then
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
						my_ss_xyz[i][j][k][6]:set(hx, hy, hz)
					else
						mpi.send(owner, i, j, k, hx, hy, hz)
					end
				end
			end
		end
	else
		--get fields back
		for i = 1,micro_count[rank] do
			i, j, k, hx, hy, hz = mpi.recv(1)
			my_ss_xyz[i][j][k][6]:set(hx, hy, hz)
		end
	end

	if rank == 1 then
		t = string.format("%08.5f", time())
		print(t)
		f = io.open(filename_prefix .. t .. ".dat", "w")
-- 		f:write("# Spin system configuration for time = " .. time() .. "\n")
		for i=1,nmacro do
			for j=1,nmacro do
				for k=1,nmacro do
					x, y, z = ss_macro:spin(i,j,k)
					ms = nmicro^3
					t = {i, j, k, x/ms, y/ms, z/ms}
					f:write(table.concat(t, "\t") .. "\n")
				end
			end
		end
		f:close()
	end
-- f:write("# this data is from the script " .. argv[2] .. "\n")
-- f:write("# Working on Tc vs J\n")
-- f:write("# time  temp  <Mx/Ms>  <My/Ms>  <Mz/Ms>  <(Mz/Ms)^2>-<(Mz/Ms)>^2\n")
-- f:flush()

end

print(name, rank, "done")

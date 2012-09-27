--rnum = 0
--for i = 1,10 do
--	psr()
--	print(seed, " : ", rnum)
--end

pi = math.pi

Ms = 550.0 			-- emu/cc
alpha = 0.5 			-- unitless
Ks = 3500000.0		-- erg/cc
gamma = 5.6*pi/1000.0	-- gyromagnetic ratio (rad/Oe 1/ns)
ax = 8.0  * (1E-7)		-- nm (1E-7 cm/nm) = cm
ay = 8.0  * (1E-7)		-- nm (1E-7 cm/nm) = cm
az = 10.0  * (1E-7)		-- nm (1E-7 cm/nm) = cm
vol = ax*ay*az		-- cc
kb = 1.38065 * (1E-16) 	-- erg/K
T = 300.0 			-- K
kT = kb*T			--erg
H0 = -1			-- Oe
dH = 50			-- Oe
HC = 4*pi
f0 = 1.0e9 			-- Hz

n = 16
ss = SpinSystem.new(n,n,2)
ss_min_cls = SpinSystem.new(n,n,2)
ss_min_far = SpinSystem.new(n,n,2)
ss_max     = SpinSystem.new(n,n,2)

mag = Magnetostatic.new(ss)
mag:setTruncation(100)
mag:setCellDimensions(ax, ay, az)
mag:setUnitCell({ax,0,0},{0,ay,0},{0,0,az})

mag:apply(ss)

-- error("AAA")

ani = Anisotropy.new(ss)
for nx=1,n do
	for ny=1,n do
		local thta, phi
		local kx,ky,kz,kmag
		thta = 0.1*pi
		phi = 0.0
		kx = math.cos(phi)*math.sin(thta)*Ks
		ky = math.sin(phi)*math.sin(thta)*Ks
		kz = math.cos(thta)*Ks
		kmag = Ks
		ani:add( {nx,ny},{kx ,ky,kz}, kmag )
		ss:setSpin({nx,ny},{0,0,1},Ms)
	end
end

zee = AppliedField.new(ss)
zee:set({0.1,0,-1000})

wood = Wood.new()

ss:resetFields()
zee:apply(ss)
ss:sumFields()

update = wood:apply(ss,ani,ss_min_cls,0)
update = wood:apply(ss,ani,ss_min_far,1)
update = wood:apply(ss,ani,ss_max,2)

for nx=1,n do
	for ny=1,n do
		local sx,sy,sz, smag
		sx,sy,sz, smag = ss:spin(nx,ny)
		print("Starting orientation: ",sx,sy,sz)
		sx,sy,sz, smag = ss_min_cls:spin(nx,ny)
		print("Closest minimum: ",sx,sy,sz)
		sx,sy,sz, smag = ss_min_far:spin(nx,ny)
		print("Furthest minimum: ",sx,sy,sz)
		sx,sy,sz, smag = ss_max:spin(nx,ny)
		print("Lowest Maximum: ",sx,sy,sz)

	end
end
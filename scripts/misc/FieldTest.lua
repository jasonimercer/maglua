-- Testing to see what the individual fields look like
eV, Tesla, ps, nm = 1, 1, 1, 1

n        = 10
uB       = 5.788321e-5 * eV / Tesla
Oe       = 1e-4 * Tesla
Gauss    = Oe
erg      = 6.24150974e11 * eV
emu      = erg/Gauss
cm       = 1e7 * nm
cc       = cm^3
a        = (10 * nm)
cell     = a^3
hbar     = 6.58211e-4 * eV * ps

Field    = 2000 * Oe
Ms       =  400 * emu/cc * cell
Ku       =  4e5 * erg/cc
A        = 0.05 * 10^-6 * erg/cm 
alpha    = 2.0
dt       = 0.1 * ps / hbar

ss      = SpinSystem.new(n, n, n)
field   = AppliedField.new(n, n, n)
ani     = Anisotropy.new(n, n, n)
ex      = Exchange.new(n, n, n)

-- setup spins 
for k=1,n do
	for j=1,n do
		for i=1,n do
			-- align spins along +Z
			ss:setSpin(i, j, k, 0, 0, Ms/uB)

			-- easy axis is Z
			ani:setSite(i, j, k, 0, 0, 1, 2*Ku*cell)

			-- periodic exchange
			J = 1.5*2*A*a*uB*uB/10

			ex:addPath(i, j, k, i+1, j,   k,   J)
			ex:addPath(i, j, k, i-1, j,   k,   J)
			ex:addPath(i, j, k, i,   j+1, k,   J)
			ex:addPath(i, j, k, i,   j-1, k,   J)
			ex:addPath(i, j, k, i,   j,   k+1, J)
			ex:addPath(i, j, k, i,   j,   k-1, J)
		end
	end
end
field:set(0,0,Field*uB)





--Test the 2 cases:
ss:zeroFields()
field:apply(ss)
ani:apply(ss)
ex:apply(ss)

print("These are the raw H values for the case of spin,")
print("applied field and anisotropy aligned along Z")

print("Exchange:   ", ss:getField("Exchange", 1, 1, 1)) --Hex at site 1,1,1
print("Anisotropy: ", ss:getField("Anisotropy", 1, 1, 1))
print("Applied:    ", ss:getField("Applied", 1, 1, 1))



-- setup spins
for k=1,n do
	for j=1,n do
		for i=1,n do
			-- align spins along +X
			ss:setSpin(i, j, k, Ms/uB, 0, 0)
		end
	end
end

ss:zeroFields()
field:apply(ss)
ani:apply(ss)
ex:apply(ss)

print("\n")
print("These are the raw H values for the case of spin along X,")
print("and applied field and anisotropy aligned along Z")

print("Exchange:   ", ss:getField("Exchange", 1, 1, 1)) --Hex at site 1,1,1
print("Anisotropy: ", ss:getField("Anisotropy", 1, 1, 1))
print("Applied:    ", ss:getField("Applied", 1, 1, 1))

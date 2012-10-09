-- maglua://CGS.lua provides the basic units for magnetism in the centimetre-gram-second system of units. The units and values provided are: cm, gram, s, Kelvin, erg, uerg, Gauss, emu, Oe, cc, nm, ps, ns, kB and gamma
--
-- Example usage: 
-- <pre>dofile("maglua://CGS.lua")
-- 
-- Ms = 1414 * emu/cc
-- cell = (8*nm) * (8*nm) * (16*nm)
-- ss = SpinSystem.new(10,10)
-- ss:setSpin({1,1}, {0,0,1}, Ms*cell)
-- </pre>

cm, gram, s, Kelvin = 1, 1, 1, 1

erg = gram*cm^2/s^2
uerg = erg*10^(-6)
Gauss = math.sqrt(gram/(cm*s^2))
emu = cm^2*math.sqrt(gram*cm)/s

Oe = Gauss

cc = cm^3
nm = 1e-7 * cm
ps = 1e-12 * s
ns = 1000 * ps
kB = 1.3806488e-16 * erg/Kelvin

gamma = 17.608597e6

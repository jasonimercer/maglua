-- Section: SpinSystem
--
-- A SpinSystem holds all data related to a micromagnetics calculation. This 
-- includes spin vectors and calculated fields as well as other quantities such
-- as time, damping and gyromagnetic ratio.
--
-- When you create a new SpinSystem, you must define a lattice size. The 
-- following creates a 3 layer 32x32 lattice and stores it in the variable
-- ss.

ss = SpinSystem.new(32, 32, 3)

-- The physical size of this lattice and boundary conditions are not considered
-- for a SpinSystem which simply holds information. Operators which operate on
-- the SpinSystem are adjusted to take into consideration different boundaries
-- and physical dimensions. 
--
-- Setting the orientation of spins in a SpinSystem is achieved with the 
-- "setSpin" method. To set the spin at location (1,1,1) to align along the +z
-- axis, one could write

ss:setSpin({1,1,1}, {0,0,1})

-- Alternatively, the same could be written as

ss:setSpin(1,1,1, 0,0,1)

-- The value of a spin at a site can be read with the "spin" method

sx, sy, sz = ss:spin({1,1,1})
print("Spin at (1,1,1): ", sx, sy, sz)

-- The time can be set and retrieved with

ss:setTime(10)
print("Current Time:", ss:time())

-- On its own, a SpinSystem simply stores data. Field operators can generate 
-- fields and LLG objects can move the system through time.

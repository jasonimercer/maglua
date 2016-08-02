mpi_cart = mpi.cart_create({3,3}, {true,true}, true)

t = mpi_cart:bcast(1, mpi_cart:gather(1, mpi_cart:get_rank()))

-- mpi.barrier()
mpi_cart:barrier()

if mpi_cart:get_rank() == 1 then
	for k,v in pairs(t) do
		print(k,v)
	end
end

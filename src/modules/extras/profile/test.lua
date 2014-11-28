function f(x)
    return x+2
end

function g(x)
    return x*x
end

print(f(4))


g(1)
g(2)
g(3)

if mpi.get_rank() == 1 then
    mpi.send(2, g, "hello")
end

if mpi.get_rank() == 2 then
    local a,b = mpi.recv(1)
    a(10)
end

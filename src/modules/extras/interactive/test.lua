
a = 4

function f()
    local caller = debug.getinfo(2)
    local d = debug.getinfo(caller.func)
    d = caller
    
    
    local a = 5
    
    interactive("interactive")
    
end

f()
print("hello")


-- checkpointSave("test.dat", function() end)
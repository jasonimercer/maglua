print([[
#include "spheres.h"
/*
typedef struct sphere
{
    const double vertex[3];
    const int* neighbours;
} sphere;
*/

]])


for n=1,5 do

dofile("face"..n..".txt")
dofile("sphere"..n..".txt")

local verts = {}

local neighbours = {}

local mapping = {}

for kf,face in pairs(f) do
    for i=1,3 do
        local idx = face[i]

        if mapping[idx] == nil then
            table.insert(verts, v[idx])

            mapping[idx] = table.maxn(verts)
        end

        local m_idx = mapping[idx]
        neighbours[m_idx] = neighbours[m_idx] or {}
        neighbours[m_idx][face[1]] = true
        neighbours[m_idx][face[2]] = true
        neighbours[m_idx][face[3]] = true
    end
end

for k,v in pairs(neighbours) do
    local n = {}
    for a,b in pairs(v) do
        table.insert(n, string.format("%4d", mapping[a]-1))
    end
    neighbours[k] = n
end


local _n = {}
local total = 0
local offsets = {}
for k,v in pairs(verts) do
    offsets[k] = total
    _n[k] = string.format("/*vert=%5d, offset=%6d*/  %s,   -1", k, total, table.concat(neighbours[k], ", "))
    total = total + table.maxn(neighbours[k]) + 1
end


-- print([[#include "spheres.h"]])


print(string.format("const int nn%d[%d] = {", n, total))
print(table.concat(_n, ",\n"))
print("};")


for k,v in pairs(verts) do
    local xyz = string.format("{% 016.15f,% 016.15f,% 016.15f}", v[1], v[2], v[3])
    local nnn = string.format("&nn%d[%4d]", n, offsets[k])
    verts[k] = "\t{" .. xyz .. ", " .. nnn .. "}"
end

print(string.format("sphere sphere%d[%d] = {", n, table.maxn(verts)+1))
print(table.concat(verts, ",\n") .. ",")
print("\t{0,0}")
print("};")


end

print([[

const sphere* get_sphere(int n)
{
    switch(n)
    {
    case 1:  return sphere1;
    case 2:  return sphere2;
    case 3:  return sphere3;
    case 4:  return sphere4;
    case 5:  return sphere5;
    }
    return 0;
}
#if 0
int main(int argc, char** argv)
{
    return 0;
}
#endif

]])

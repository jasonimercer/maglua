-- 
-- Sample script showing use of "checkpointToString" and "checkpointFromString"
-- 
-- These functions will save and load data using the uuencode/uudecode algorithm.
-- Encoded data use only printable characters and so can be easily emailed, 
-- copied to a clipboard, stored in a text file or stored in a database.
-- 

s = checkpointToString({"a"}, function() return "hello" end, math.pi)

print(s)

x,y,z = checkpointFromString(s)

print(x[1],y(),z)

-- Script Output:
--[[
begin 600 checkpoint.dat
M0TA%0TM03TE.5```````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M```````````````````````````````````````````````````#````'@``
M``4````!`````P```````````/`_!`````(```!A`'\````&````=P```!M,
M=6%1``$$"`0(`!@`````````0'5U7V-H96-K<&]I;G1?=&5S="YL=6$`"0``
M``D````````"`P````$````>```!'@"```$````$!@````````!H96QL;P``
K`````P````D````)````"0``````````````#`````,````8+414^R$)0```
`
end

a	hello	3.1415926535898

]]

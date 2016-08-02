-- Fingerprint Module

local orig_dofile = dofile

Fingerprint = {}

local functions = {}

functions["hash128"] = {"Generate a 128 bit MurMur hash of the supplied string.",
                      "1 String: Data to hash.",
                      "1 String: 32 hexadecimal characters representing the 128 bit MurMur hash of the input.",
                      fp_hash128}

functions["hash32"]  = {"Generate a 32 bit MurMur hash of the supplied string.",
                      "1 String: Data to hash.",
                      "1 String: 8 hexadecimal characters representing the 32 bit MurMur hash of the input.",
                      fp_hash32}

functions["hashToArt"]  = {[[Generate a Visual Key of a given hexadecimal input. Example 
<pre>
hash  = "78A3DC"
print(Fingerprint.hashToArt(hash, "title", 10, 5))
</pre>Output:
<pre>
+--[title]--+
|E.         |
|  . .      |
|   o S     |
|  . = .    |
|   o .     |
+-----------+
</pre>]],
                      "1 String, 1 optional string, 2 optional integers: Hexadecimal string. Optional title, optional width and height.",
                      "1 String: Ascii-art of the visual key.",
                      fp_hashFingerprint}



Fingerprint.hash128 = fp_hash128
Fingerprint.hash32 = fp_hash32
Fingerprint.hashToArt = fp_hashFingerprint

local function Split(str, delim, maxNb)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    if maxNb == nil or maxNb < 1 then
        maxNb = 0    -- No limit
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 0
    local lastPos
    for part, pos in string.gfind(str, pat) do
        nb = nb + 1
        result[nb] = part
        lastPos = pos
        if nb == maxNb then break end
    end
    -- Handle the last field
    if nb ~= maxNb then
        result[nb + 1] = string.sub(str, lastPos)
    end
    return result
end

local isFingerprinted = function(data)
    local lines = Split(data, "\n", 13)

    local l1 = string.find((lines[1] or ""), "^-- This is a Fingerprinted script")
    local l3, _, hash = string.find((lines[3] or ""), "^-- MurmurHash128:%s*(%S+)")
    local l11 = string.find((lines[11] or ""), "^--.*allows humans to quickly identify mismatches%.")

    if l1 and l3 and l11 then
        local len = 0
        for i=1,11 do
            len = len + string.len(lines[i] or "") + 1
        end

        return true, hash, string.sub(data, len+1)
    end
    return false
end

functions["isFingerprinted"]  = {"Decide if a given string input represents a Fingerprinted script.",
                      "1 String: Test string",
                      "1 Boolean, 2 Strings or 2 nils: True if it the input is Fingerprinted plus the hash in the fingerprint and the body of the script. Otherwise false, nil, nil.",
                      isFingerprinted}


local addHeader = function(data)
    local hash = Fingerprint.hash128(data)
    
    local output = {
        "-- This is a Fingerprinted script.",
        "-- ",
        "-- MurmurHash128: " .. hash,
        "-- ",
        "-- If the Fingerprint module is installed, a Fingerprinted",
        "-- script will not run if the body has been changed since",
        "-- the fingerprint was generated. Deleting these 11 comment",
        "-- lines will remove this restriction or you may reprint",
        "-- the file with the --refingerprint command line option.",
        "-- The visual representation of the hash to the right",
        "-- allows humans to quickly identify mismatches."
    }

    local b = 8
    local art = Fingerprint.hashToArt(hash, "Murmur128", b*2+1, b+1)

    local lines_art = Split(art, "\n")
    for k,v in pairs(lines_art) do
        output[k] = (output[k] or "--") .. string.rep(" ", 80)
        output[k] = string.sub(output[k], 1, 61)
        output[k] = output[k] .. lines_art[k]
    end


    return table.concat(output, "\n") .. "\n" .. data
end

functions["addHeader"]  = {"Add a Fingerprint header to a string. If this string is writted to file and the Fingerprint module is installed, the file is considered locked. A locked file cannot have it's body changed without updating the header. Files with header-body mismatches will not execute in MagLua.",
                      "1 String: Body to add a fingerprint.",
                      "1 String: Input with Fingerprint header prepended.",
                      addHeader}



fp_hashFingerprint = nil
fp_hash128 = nil
fp_hash32 = nil



function dofile(x)
    if Fingerprint.ignore == true then
        return orig_dofile(x)
    end

    f, msg = io.open(x, "r")

    if f == nil then
        return orig_dofile(x)
    end

    local content = f:read("*all")

    f:close()
    
    local fp, hash, body = Fingerprint.isFingerprinted(content)

    if fp then
        if Fingerprint.hash128(body) ~= hash then
            error("Fingerprint: Hash mismatch.\nThe body of `" .. x .. "' has been changed since the fingerprint was generated. Scripts are fingerprinted so that they do not get changed. If you need to run the script you may remove the fingerprint, --refingerprint, --ignore-fingerprints or remove the Fingerprint module.", 3)
        end
    end

    return orig_dofile(x)
end

for k,v in pairs(functions) do
    Fingerprint[k] = v[4]
end

Fingerprint.help = function(x)

    for k,v in pairs(functions) do
        if x == v[4] then
            return v[1], v[2], v[3]
        end
    end

    if x == nil then
        return [[The Fingerprint module allow Fingerprint headers with hashes of the body to be generated and prepended to scripts. It augments the dofile function and dissallows the execution of scripts which have header-body mismatches. This module adds the following command line switches:
<pre>
 --fingerprint   [infile|-] [outfile|-]       Print the infile or stdin with a fingerprint header
                                              to the outfile or stdout.

 --refingerprint [infile|-] [outfile|-]       Reprint the infile or stdin with a fingerprint header
                                              to the outfile or stdout.

 --remove-fingerprint [infile|-] [outfile|-]  Print the infile or stdin with a fingerprint header
                                              removed to the outfile or stdout.

 --ignore-fingerprints                        Ignore fingerprint restrictions.
</pre>
Here is an example of a small script with a Fingerprint header:<pre>
-- This is a Fingerprinted script.                           +---[Murmur128]---+
--                                                           |E....            |
-- MurmurHash128: 5BC8B36FEE4F3032AC8AD4ED89494D46           |     .           |
--                                                           |      .          |
-- If the Fingerprint module is installed, a Fingerprinted   |     . o .       |
-- script will not run if the body has been changed since    |      o S +      |
-- the fingerprint was generated. Deleting these 11 comment  |   . = . B o     |
-- lines will remove this restriction or you may reprint     |  . o + o   .    |
-- the file with the --refingerprint command line option.    | . o = . ...     |
-- The visual representation of the hash to the right        |  . + o  ++..    |
-- allows humans to quickly identify mismatches.             +-----------------+
for i=1,10 do
    print("Hello", i)
end
</pre>
If the "Hello" is changed to "Hello!" and the script is run, the following error will be generated:<pre>
Fingerprint: Hash mismatch.
The body of `example.lua' has been changed since the fingerprint was generated. Scripts are fingerprinted 
so that they do not get changed. If you need to run the script you may remove the fingerprint, 
--refingerprint, --ignore-fingerprints or remove the Fingerprint module.
</pre>
]]
    end

    return help(x)
end

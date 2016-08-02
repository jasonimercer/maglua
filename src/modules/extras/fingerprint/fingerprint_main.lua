-- In this file we will add command line help switches and code

help_args = help_args or {}

table.insert(help_args, {"",""})
table.insert(help_args, {"Fingerprint Related:",""})
table.insert(help_args, {"--fingerprint   [infile|-] [outfile|-]", "Print the infile or stdin with a fingerprint header"})
table.insert(help_args, {"","to the outfile or stdout."})

table.insert(help_args, {"--refingerprint [infile|-] [outfile|-]", "Reprint the infile or stdin with a fingerprint header"})
table.insert(help_args, {"","to the outfile or stdout."})

table.insert(help_args, {"--remove-fingerprint [infile|-] [outfile|-]", "Print the infile or stdin with a fingerprint header"})
table.insert(help_args, {"","removed to the outfile or stdout."})

table.insert(help_args, {"--ignore-fingerprints", "Ignore fingerprint restrictions."})

for i,v in ipairs(arg) do
    if v == "--ignore-fingerprints" then
        Fingerprint.ignore = true
    end
end
    
for i,v in ipairs(arg) do
    if v == "--fingerprint" then
	local p = arg[i+1]
	local q = arg[i+2]
	if p and q then
            local f = nil
            local fout = nil

            if p == "-" then
                f = io.stdin
            else
                f = io.open(p, "r")
            end
            if f == nil then
                error("Failed to read " .. tostring(p))
            end

            if q == "-" then
                fout = io.stdout
            else
                fout = io.open(q, "w")
                if fout == nil then
                    error("Failed to open " .. tostring(q) .. " for writing")
                end
            end

            local content = f:read("*all")

            if p ~= "-" then
                f:close()
            end

            local fp, hash, body = Fingerprint.isFingerprinted(content)
            
            if fp then
                if Fingerprint.hash128(body) ~= hash then
                    error("Data contains fingerprint which does not match body. Consider using --refingerprint")
                else
                    fout:write(content)
                end
            else
                fout:write(Fingerprint.addHeader(content))
            end
            shutdown_now = true
            if fout ~= io.stdout then
                fout:close()
            end
        else
            error("--fingerprint requires 2 arguments")
	end
	table.remove(arg, i+2)
	table.remove(arg, i+1)
	table.remove(arg, i)
    end


    if v == "--refingerprint" then
	local p = arg[i+1]
	local q = arg[i+2]
	if p and q then
            local f = nil
            local fout = nil
            if p == "-" then
                f = io.stdin
            else
                f = io.open(p, "r")
            end
            if f == nil then
                error("Failed to read " .. tostring(p))
            end

            if q == "-" then
                fout = io.stdout
            else
                fout = io.open(q, "w")
                if fout == nil then
                    error("Failed to open " .. tostring(q) .. " for writing")
                end
            end

            local content = f:read("*all")

            if p ~= "-" then
                f:close()
            end

            local fp, hash, body = Fingerprint.isFingerprinted(content)
            
            if fp then
                if Fingerprint.hash128(body) ~= hash then
                    fout:write(Fingerprint.addHeader(body))
                else
                    fout:write(content)
                end
            else
                fout:write(Fingerprint.addHeader(content))
            end
            shutdown_now = true
            
            if fout ~= io.stdout then
                fout:close()
            end
        else
            error("--refingerprint requires 2 arguments")
	end
	table.remove(arg, i+2)
	table.remove(arg, i+1)
	table.remove(arg, i)
    end


    if v == "--remove-fingerprint" then
	local p = arg[i+1]
        local q = arg[i+2]
	if p and q then
            local f = nil
            local fout = nil
            if p == "-" then
                f = io.stdin
            else
                f = io.open(p, "r")
            end
            if f == nil then
                error("Failed to read " .. tostring(p))
            end

            if q == "-" then
                fout = io.stdout
            else
                fout = io.open(q, "w")
                if fout == nil then
                    error("Failed to open " .. tostring(q) .. " for writing")
                end
            end

            local content = f:read("*all")

            if p ~= "-" then
                f:close()
            end

            local fp, hash, body = Fingerprint.isFingerprinted(content)
            
            if fp then
                fout:write(body)
            else
                fout:write(content)
            end
            if fout ~= io.stdout then
                fout:close()
            end
            shutdown_now = true
        else
            error("--remove-fingerprint requires 2 arguments")
	end
	table.remove(arg, i+2)
	table.remove(arg, i+1)
	table.remove(arg, i)
    end
end


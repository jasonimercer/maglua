-- this file contains functions related to collecting
-- information about magnetization

function resetMagStats()
	-- print("reset")
	mags = {}
	for k,v in pairs(regions) do
		mags[k] = {} --will insert x,y,z,m tuples
	end
end

function collectMagStats()
	-- print("collect")
	for k1,r in pairs(regions) do
		local x, y, z, n = 0, 0, 0, 0, 0
		for k2,s in pairs(r) do
			local sx, sy, sz = ss:spin(s.x, s.y, s.z)
			x = x + sx
			y = y + sy
			z = z + sz
			n = n + 1
		end
		if n == 0 then
			n = 1
		end
-- 		print("Averaging over " .. n .. " datapoints")	
		table.insert(mags[k1], 
				{x/n, y/n, z/n, 
				((x^2+y^2+z^2)^(1/2))/n})
	end
end

function calculateMagStats()
	-- print("calc")
	local stats = {}
	for r,t in pairs(mags) do
		local sum = {0,0,0,0}
		local count = 0
		for k,v in pairs(t) do
			for c = 1,4 do --component: x,y,z,m
				sum[c] = sum[c] + v[c]
			end
			count = count + 1
		end
		if count == 0 then
			count = 1
		end
		stats[r] = {}
		for c = 1,4 do --component: x,y,z,m
			stats[r][c] = sum[c]/count
		end
	end
	return stats
end

-- report the statistics in cols col to file f
function reportMagStats(stats, cols, f)
	local line = {}
	for i,v in ipairs(cols) do
		table.insert(line, table.concat(stats[v], "\t"))
	end
	f:write(table.concat(line, "\t") .. "\n")
	-- f:flush()
end

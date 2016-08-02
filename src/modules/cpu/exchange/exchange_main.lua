-- In this file we will add command line help switches
-- controlling Exchange.defaultMomentNormalization

help_args = help_args or {}

table.insert(help_args, {"",""})
table.insert(help_args, {"Exchange Related:",""})
table.insert(help_args, {"--exchange-set-normalization X", "Set default normalization flag."})

for i,v in ipairs(arg) do
    if v == "--exchange-set-normalization" then
	local p = arg[i+1]
	if p then
            Exchange = Exchange or {}
            Exchange.defaultMomentNormalization = loadstring("return " .. p)()
	    table.remove(arg, i+1)
        else
            error("--exchange-set-normalization requires an argument")
	end
	table.remove(arg, i)
    end
end


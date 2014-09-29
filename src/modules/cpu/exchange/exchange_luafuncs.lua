-- Exchange
local MODNAME = "Exchange"
local MODTAB = _G[MODNAME]
local t = maglua_getmetatable(MODNAME) -- this is a special function available only at registration time


local ex_new = Exchange.new

-- reading default moment normalization from Exchange.defaultMomentNormalization
Exchange.new = function(...)
                   local ex = ex_new(...)
                   ex:normalizeMoments(Exchange.defaultMomentNormalization)
                   if Exchange.defaultMomentNormalization ~= nil then
                       ex:_disable_warning()
                   end
                   return ex
               end
 
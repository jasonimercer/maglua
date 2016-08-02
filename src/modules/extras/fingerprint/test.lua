x = "abcdefg1234" .. (arg[1] or "")

if false then
    h = Fingerprint.hash32(x)

    --print(x)
    --print(h)
    
    fp = Fingerprint.hashToArt(h)
    --print(fp)
    
    
    fp = Fingerprint.hashToArt(h, "Murmur 32")
    --print(fp)
end

d = Fingerprint.addHeader(x)

print(d)

local fp, hash, body = Fingerprint.isFingerprinted(d)

print(fp)
if fp then
    print(hash)
    print(Fingerprint.hash128(body))
end

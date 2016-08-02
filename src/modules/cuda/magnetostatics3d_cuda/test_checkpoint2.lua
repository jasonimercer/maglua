mag, mag2, ss = checkpointLoad("foo.dat")


print(mag:internalData())

mag2:apply(ss)

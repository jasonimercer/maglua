function landscape_plot(arg)
	local filename = "energy_landscape.txt"
	local ss2 = ss:copy()
	local f = io.open(filename, "w")
	
	local dt=math.pi/128
	for t1=0,math.pi+dt/2,dt do
		local d1 = {math.sin(t1), 0, math.cos(t1)}
		local _,_,_,m1 = ss2:spin({1,1,1})
		ss2:setSpin({1,1,1}, d1, m1)
		for t2=0,math.pi+dt/2,dt do
			local d2 = {math.sin(t2), 0, math.cos(t2)}
			local _,_,_,m2 = ss2:spin({1,1,2})
			ss2:setSpin({1,1,2}, d2, m2)
			f:write(table.concat({t1,t2,energy(ss2)}, "\t") .. "\n")
		end
		f:write("\n")
	end
	f:close()
	
	
	extra_plots = ""
	for k,v in pairs(arg) do
		extra_plots = extra_plots .. string.format([[, "%s" w lp lt %s lw 1.5]], v, k)
	end
	
	
-- 	extra_plots =  , "FinalComponents.dat" w l lt 3 lw 1.5 ]]
	
		
	cmd = string.format([[
#		set term wxt size 1280,960
		set term svg size 1280,960 enhanced fname "Times" fsize 14 lw 1
        set output "plot2.svg"
		set xrange [0:pi]
		set yrange [0:pi]
		set xlabel "Theta 1"
		set ylabel "Theta 2"
		
		set xrange [0:pi]
		set yrange [0:pi]
		set isosample 250, 250
		set table 'test.dat'
		splot "./energy_landscape.txt"
		unset table
		
		
		set contour base
		#set cntrparam level incremental -130, 0.33, 100
		set cntrparam levels 50
		#set cntrparam levels discrete -126,-122,-121,-120,-119.5,-119,-118,-114,-110
		unset surface
		set table "cont.dat"
		splot "./energy_landscape.txt"
		unset table
		
		reset
		set xrange [0:pi]
		set yrange [0:pi]
		unset key
		#set palette defined ( 0 0 0 1, 0.5 1 1 1, 1 1 0 0 )
		set palette rgbformulae 33,13,10
		
		l '<./cont.sh cont.dat 0 15 0'

		
		plot 'energy_landscape.txt' with image, '<./cont.sh cont.dat 1 15 0' w l lt -1 lw 1.5 %s

		]], extra_plots)

	f = io.open("plot2.cmd", "w")
	f:write(cmd)
	f:close()

	os.execute("gnuplot plot2.cmd && convert plot2.svg MAP.png && rm plot2.cmd plot2.svg")
end

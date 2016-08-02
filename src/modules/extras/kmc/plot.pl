#!/usr/bin/perl

open(G, "| gnuplot");

print G qq!
set term png
set output "KMC.png"
set xlabel "Time"
set xtics 5e-7
set ylabel "<Mz>"
set key bottom left
plot "wood_example_kmc/KMC_new_data.dat" using 1:3 w l title "New Framework", "wood_example/M_data.dat" using 1:3 w l title "Old Results"
!;

close(G);

print `scp KMC.png cmms01.ace-net.ca:~jmercer/public_html`;

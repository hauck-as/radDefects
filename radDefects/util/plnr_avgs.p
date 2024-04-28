#!/usr/bin/gnuplot -persist
# Gnuplot script file for plotting data in sxdefectalign "vline-eV-a{0,1,2}.dat" files
# This file is called    plnr_avgs.p
set terminal png size 1600,600
set autoscale
unset log
unset label
set xtic auto
set ytic auto
set ylabel "potential [eV]"
set output "vline-eV.png"
set multiplot layout 1,3 columns title "Planar Averaged Potentials"
unset title
unset key
set size 0.27,0.95
set origin 0.03,0.0
set lmargin 4
set rmargin 4
set xlabel "a [bohr]"
plot "vline-eV-a0.dat" u 1:2 t 'V_{lr}' w lines dashtype 4 lw 2, \
"vline-eV-a0.dat" u 1:3 t 'V_{defect} - V_{ref}' w lines dashtype 1 lw 2, \
"vline-eV-a0.dat" u 1:4 t 'V_{defect} - V_{ref} - V_{lr}' w lines dashtype 12 lw 2
set size 0.27,0.95
set origin 0.295,0.0
set lmargin 6
set rmargin 2
set xlabel "b [bohr]"
plot "vline-eV-a1.dat" u 1:2 t 'V_{lr}' w lines dashtype 4 lw 2, \
"vline-eV-a1.dat" u 1:3 t 'V_{defect} - V_{ref}' w lines dashtype 1 lw 2, \
"vline-eV-a1.dat" u 1:4 t 'V_{defect} - V_{ref} - V_{lr}' w lines dashtype 12 lw 2
set size 0.415,0.95
set origin 0.585,0.0
set xlabel "c [bohr]"
set lmargin 4
set rmargin 24
set key rmargin
plot "vline-eV-a2.dat" u 1:2 t 'V_{lr}' w lines dashtype 4 lw 2, \
"vline-eV-a2.dat" u 1:3 t 'V_{defect} - V_{ref}' w lines dashtype 1 lw 2, \
"vline-eV-a2.dat" u 1:4 t 'V_{defect} - V_{ref} - V_{lr}' w lines dashtype 12 lw 2
unset multiplot
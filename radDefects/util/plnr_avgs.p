#!/usr/bin/gnuplot -persist
# Gnuplot script file for plotting data in sxdefectalign "vline-eV-a{0,1,2}.dat" files
# This file is called    plnr_avgs.p
set terminal png size 1400,500
set autoscale
unset log
unset label
set xtic auto
set ytic auto
set ylabel "Potential (V)"
set output "vline-eV.png"
set multiplot layout 1,3 columns title "Planar Averaged Potentials"
unset title
unset key
set xlabel "a (Bohr)"
plot "vline-eV-a0.dat" u 1:2 t 'V(long-range)' w lines, \
"vline-eV-a0.dat" u 1:3 t 'V(defect)-V(ref)' w lines, \
"vline-eV-a0.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w lines
set xlabel "b (Bohr)"
plot "vline-eV-a1.dat" u 1:2 t 'V(long-range)' w lines, \
"vline-eV-a1.dat" u 1:3 t 'V(defect)-V(ref)' w lines, \
"vline-eV-a1.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w lines
set xlabel "c (Bohr)"
set key rmargin
plot "vline-eV-a2.dat" u 1:2 t 'V(long-range)' w lines, \
"vline-eV-a2.dat" u 1:3 t 'V(defect)-V(ref)' w lines, \
"vline-eV-a2.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w lines
unset multiplot
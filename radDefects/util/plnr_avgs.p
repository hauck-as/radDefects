#!/usr/bin/gnuplot -persist
# Gnuplot script file for plotting data in sxdefectalign "vline-eV-a{0,1,2}.dat" files
# This file is called    plnr_avgs.p
set terminal png size 1600,600
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
set size 0.26,0.95
set origin 0.03,0.0
set lmargin 4
set rmargin 4
set xlabel "a (Bohr)"
plot "vline-eV-a0.dat" u 1:2 t 'V(long-range)' w lines, \
"vline-eV-a0.dat" u 1:3 t 'V(defect)-V(ref)' w lines, \
"vline-eV-a0.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w lines
set size 0.26,0.95
set origin 0.285,0.0
set lmargin 6
set rmargin 2
set xlabel "b (Bohr)"
plot "vline-eV-a1.dat" u 1:2 t 'V(long-range)' w lines, \
"vline-eV-a1.dat" u 1:3 t 'V(defect)-V(ref)' w lines, \
"vline-eV-a1.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w lines
set size 0.44,0.95
set origin 0.565,0.0
set xlabel "c (Bohr)"
set lmargin 4
set rmargin 32
set key rmargin
plot "vline-eV-a2.dat" u 1:2 t 'V(long-range)' w lines, \
"vline-eV-a2.dat" u 1:3 t 'V(defect)-V(ref)' w lines, \
"vline-eV-a2.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w lines
unset multiplot
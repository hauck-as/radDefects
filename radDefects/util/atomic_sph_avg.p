#!/usr/bin/gnuplot -persist
# Gnuplot script file for plotting data in sxdefectalign "vAtoms.dat" files
# This file is called    atomic_sph_avg.p
set terminal png size 1200,600
set autoscale
unset log
unset label
set xtic auto
set ytic auto
set title "Atomic Sphere Averages"
set key rmargin
set xlabel "r (Bohr)"
set ylabel "Potential (V)"
set output "vAtoms.png"
plot "vAtoms.dat" u 1:2 t 'V(long-range)' w points, \
"vAtoms.dat" u 1:3 t 'V(defect)-V(ref)' w points, \
"vAtoms.dat" u 1:4 t 'V(defect)-V(ref)-V(long-range)' w points
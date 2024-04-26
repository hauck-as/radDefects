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
set rmargin 32
set key rmargin
set xlabel "r [bohr]"
set ylabel "potential [eV]"
set output "vAtoms.png"
plot "vAtoms.dat" i 0 u 1:2 t 'V(long-range)' w points, \
"vAtoms.dat" i 0 u 1:3 t 'V(defect)-V(ref)' w points, \
"vAtoms.dat" i 0 u 1:4 t 'V(defect)-V(ref)-V(long-range)' w points, \
"vAtoms.dat" i 1 u 1:2 t 'V(long-range)' w points, \
"vAtoms.dat" i 1 u 1:3 t 'V(defect)-V(ref)' w points, \
"vAtoms.dat" i 1 u 1:4 t 'V(defect)-V(ref)-V(long-range)' w points
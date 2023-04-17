#!/bin/bash

exec=./build/release-gcc-10-conan-flexiblas-native/tensorbench
types=[fp64]
mpsbonds=[32,64,128,256]
mpiflags="--bind-to socket --report-bindings -x OMP_PLACES=sockets -x OMP_PROC_BIND=master"
fname="tbdb.h5"

$exec -i 10 --facc=REPLACE --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[cute]  --nomps=[1]
$exec -i 10 --facc=READWRITE --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[eigen1,tblis,xtensor]  --nomps=[1,2,4,8,16]
mpirun -n 1 $mpiflags $exec --facc=READWRITE  -i 10 --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[cyclops] --types=$types --nomps=[1]
mpirun -n 2 $mpiflags $exec --facc=READWRITE  -i 10 --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[cyclops] --nomps=[1]
mpirun -n 4 $mpiflags $exec --facc=READWRITE  -i 10 --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[cyclops] --nomps=[1]
mpirun -n 8 $mpiflags $exec --facc=READWRITE  -i 10 --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[cyclops] --nomps=[1]
mpirun -n 16 $mpiflags $exec --facc=READWRITE  -i 10 --mpsbonds=$mpsbonds -D 2 -M 14 --fname=$fname --modes=[cyclops] --nomps=[1]


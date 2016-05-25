#!/bin/sh

sx=3
sy=-2

t1s=100
t2s=1
sps=1
sts=1

t1=$t1s
t2=$t2s
sp=$sps
st=$sts

echo "\nTest 1: Variation of initialTemp"
echo "-------------------------------\n"
t1=200
while [ $t1 -gt $t2s ]
do
    echo "initialTemp = $t1, finalTemp = $t2, coolingSpeed = $sp, stepSize = $st"
    ./simulatedAnnealing.py --startx $sx --starty $sy --qbox 0 --runs 100 \
                            --t1 $t1 --t2 $t2 --speed $sp --step $st
    t1=`expr $t1 - 10`
    echo "\n"
done
t1=$t1s

echo "\nTest 2: Variation of finalTemp"
echo "-------------------------------\n"
t1=200
while [ $t2 -lt $t1s ]
do
    echo "initialTemp = $t1, finalTemp = $t2, coolingSpeed = $sp, stepSize = $st"
    ./simulatedAnnealing.py --startx $sx --starty $sy --qbox 0 --runs 100 \
                            --t1 $t1 --t2 $t2 --speed $sp --step $st
    t2=`expr $t2 + 10`
    echo "\n"
done
t1=$t1s
t2=$t2s

echo "\nTest 3: Temperature shift (dT = 100)"
echo "-------------------------------\n"
i=0
while [ $i -lt 10 ]
do
    echo "initialTemp = $t1, finalTemp = $t2, coolingSpeed = $sp, stepSize = $st"
    ./simulatedAnnealing.py --startx $sx --starty $sy --qbox 0 --runs 100 \
                            --t1 $t1 --t2 $t2 --speed $sp --step $st
    i=`expr $i + 1`
    t1=`expr $t1 + 100`
    t2=`expr $t2 + 100`
    echo "\n"
done
t1=$t1s
t2=$t2s

echo "\nTest 4: Variation of coolingSpeed"
echo "-------------------------------\n"
for var in 0.01 0.025 0.05 0.1 1 5 10 15 20
do
    echo "initialTemp = $t1, finalTemp = $t2, coolingSpeed = $var, stepSize = $st"
    ./simulatedAnnealing.py --startx $sx --starty $sy --qbox 0 --runs 100 \
                            --t1 $t1 --t2 $t2 --speed $var --step $st
    echo "\n"
done

echo "\nTest 5: Variation of stepSize, coolingSpeed = 1"
echo "-------------------------------\n"
for var in 0.1 0.5 1 1.5 2 5 10 20
do
    echo "initialTemp = $t1, finalTemp = $t2, coolingSpeed = $sp, stepSize = $var"
    ./simulatedAnnealing.py --startx $sx --starty $sy --qbox 0 --runs 100 \
                            --t1 $t1 --t2 $t2 --speed $sp --step $var
    echo "\n"
done

echo "\nTest 6: Variation of stepSize, coolingSpeed = 0.1"
echo "-------------------------------\n"
sp=0.1
for var in 0.1 0.5 1 1.5 2 5 10 20
do
    echo "initialTemp = $t1, finalTemp = $t2, coolingSpeed = $sp, stepSize = $var"
    ./simulatedAnnealing.py --startx $sx --starty $sy --qbox 0 --runs 100 \
                            --t1 $t1 --t2 $t2 --speed $sp --step $var
    echo "\n"
done
sp=$sps

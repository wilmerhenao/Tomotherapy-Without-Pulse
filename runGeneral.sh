
for ((v = 100; v <= 8000; m+=200)); do
for ((c = 3; c <= 10; c++)); do
    python tomotherapyIterAMPLGeneral.py 10 $v 40 $c
done
done

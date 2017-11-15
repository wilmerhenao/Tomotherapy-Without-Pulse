for ((v = 200; v <= 10000; v+=200)); do
for ((i = 8; i <= 45; i++)); do
    python tomotherapyIterAMPLGeneral.py 10 $v $i
done
done
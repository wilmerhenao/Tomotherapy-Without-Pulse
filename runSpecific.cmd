echo Run simple IMRT
REM C:\Intel\python\intelpython3\python multiTool.py 0.02 0.077 1 0 0 0 51 0 
echo Run IMRT with 20 msec constraint
REM C:\Intel\python\intelpython3\python multiTool.py 0.02 0.077 1 1 0 0 51 0
echo pairs Problem 9000
REM C:\Intel\python\intelpython3\python OrganizedmultiTool.py 0.02 0.170 0 0 1 0 51 0 9000
echo pairs Problem Full
REM C:\Intel\python\intelpython3\python OrganizedmultiTool.py 0.02 0.170 0 0 1 0 51 0
echo full Problem 9000
REM C:\Intel\python\intelpython3\python OrganizedmultiTool.py 0.02 0.170 0 0 0 0 51 1 9000
echo full Problem Full
C:\Intel\python\intelpython3\python OrganizedmultiTool.py 0.02 0.170 0 0 0 0 51 1
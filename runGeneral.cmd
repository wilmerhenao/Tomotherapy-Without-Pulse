(FOR /L %%M IN (5, 5, 70) DO (
FOR /L %%A IN (5, 1, 13) DO (
    echo %%M
    echo %%A
    C:\Intel\python\intelpython3\python tomoAverageOnlyTwoProjectionsNeighbors.py %%M %%A 1500
)
)
)

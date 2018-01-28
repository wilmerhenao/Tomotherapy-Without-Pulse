# -------------------------------------------------------------------
## The CMP
# -------------------------------------------------------------------
# Set definitions
param numvoxels integer > 0;
param oldobj >= 0;
param newobj >= 0;
param timeflagold >= 0;
param proctime >= 0;
param DPtimekeeper >= 0;
param CMPtimekeeper >= 0;
param numberOfIterations integer > 0;
param numProjections > 0;
param U > 0;
param realU binary;
param numLeaves integer > 0;
param projPerLoop > 0;
param numLoops integer > 0;
param MBar integer > 0;
param LBar integer > 0;
param consecutive integer > 0;

set PROJECTIONS = {0..numProjections - 1};
set PROJECTIONSM1 = {0..numProjections - 2};
set PROJECTIONSMROLL = {0..(numProjections - consecutive - 2)};
set LEAVES = {0..(numLeaves - 1)};
set LOOPS = {0..numLoops};
set VOXELS;
set KNJMPARAMETERS within {n in LEAVES, k in PROJECTIONS, j in VOXELS};
set POSSIBLEPL within {k in PROJECTIONSM1, m in LOOPS};

# Parameters
param D {KNJMPARAMETERS} >= 0;
param thethreshold {VOXELS} >= 0;
param quadHelperOver {VOXELS} >= 0;
param quadHelperUnder {VOXELS} >= 0;
param yBar>= 0, <= U;
param tprime >= 0;
param tdoubleprime >= 0;
param sbar >= 0;


# Variables
var betas {n in LEAVES, k in PROJECTIONS} binary;
var t{n in LEAVES, k in PROJECTIONSM1} binary;
var y{k in PROJECTIONS} >= 0, <= U;
var ystable >= 0, <= U;
var z {j in VOXELS} >= 0;
var z_plus {j in VOXELS} >= 0;
var z_minus {j in VOXELS} >= 0;

# Objective
minimize ObjectiveFunction: sum {j in VOXELS} (quadHelperUnder[j] * z_minus[j] * z_minus[j] + quadHelperOver[j] * z_plus[j] * z_plus[j]);

# Constraints where intensity is variable per projection
subject to doses_to_j {j in VOXELS}: z[j] = yBar * sum{ (n,k,j) in KNJMPARAMETERS}( D[n,k,j] * t[n,k]);
# Constraints with stable y
subject to positive_only {j in VOXELS}: z_plus[j] - z_minus[j] = z[j] - thethreshold[j];
# Individual Constraints
subject to indtimebig {l in LEAVES, p in PROJECTIONSM1}: tprime * betas [l, p] <= t[l, p];
subject to indtimesmall {l in LEAVES, p in PROJECTIONSM1}: t[l, p] <= delta * betas[l,p] / sbar;
subject to avgtime: sum{l in LEAVES, p in PROJECTIONSM1} t[l, p] >= tdoubleprime * sum{l in LEAVES, p in PROJECTIONSM1} betas[l, p] ;
tPerLeaf {j in VOXELS}: sum{k in PROJECTIONSM1, n in LEAVES: (n,k,j) in KNJMPARAMETERS} mu[n, k] <= 3 * 80/5;
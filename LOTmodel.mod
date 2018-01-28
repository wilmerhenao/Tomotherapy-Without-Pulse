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
param betasparam{n in LEAVES, k in PROJECTIONS} binary;
param yparam := 500;
param tprime := 0.0;
param t2prime := 0.0;

# Variables
var betas {n in LEAVES, k in PROJECTIONS} binary;
var ystable >= 0, <= U;
var z {j in VOXELS} >= 0;
var t {n in LEAVES, k in PROJECTIONS} >= 0;
var z_plus {j in VOXELS} >= 0;
var z_minus {j in VOXELS} >= 0;

# Objective
minimize ObjectiveFunction: sum {j in VOXELS} (quadHelperUnder[j] * z_minus[j] * z_minus[j] + quadHelperOver[j] * z_plus[j] * z_plus[j]);
subject to positive_only {j in VOXELS}: z_plus[j] - z_minus[j] = z[j] - thethreshold[j];

# -------------------------------------------------------------------
## The DP       
# -------------------------------------------------------------------

subject to doses_to_j_yparam {j in VOXELS}: z[j] = sum{ (n,k,j) in KNJMPARAMETERS}( D[n,k,j] * betas[n,k] * yparam);

subject to upperlimit {n in LEAVES, k in PROJECTIONS}: t[n,k] <= betas[n,k];
subject to lowerlimitaverage: sum{n in LEAVES, k in PROJECTIONS} t[n,k] >= sum{n in LEAVES, k in PROJECTIONS} t2prime * betas[n,k];
subject to lowerlimit {n in LEAVES, k in PROJECTIONS}: t[n,k] >= tprime * betas[n,k];
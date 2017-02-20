# -------------------------------------------------------------------
## The CMP
# -------------------------------------------------------------------
# Set definitions
param numvoxels integer > 0;
param numloops integer >= 0;
param oldobj >= 0;
param newobj >= 0;
param timeflagold >= 0;
param proctime >= 0;
param PPtimekeeper >= 0;
param CMPtimekeeper >= 0;
param numberOfIterations integer > 0;

set PROJECTIONS = {0..numloops * 178 - 1};
set PROJECTIONSM1 = {0..numloops * 178 - 2};
set LEAVES = {0..79};
set VOXELS;
set KNJPARAMETERS within {n in LEAVES, k in PROJECTIONS, j in VOXELS};

# Parameters
param D {KNJPARAMETERS} >= 0;
param Mbar = numloops * 50;
param U = 50;
param thethreshold {VOXELS} >= 0;
param quadHelperOver {VOXELS} >= 0;
param quadHelperUnder {VOXELS} >= 0;
param betasparam{n in LEAVES, k in PROJECTIONS} binary;
param yparam{k in PROJECTIONS} >= 0, <= U;
param yBar{k in PROJECTIONS} >= 0, <= U;

# Variables
var betas {n in LEAVES, k in PROJECTIONS} binary;
var mu{n in LEAVES, k in PROJECTIONSM1} binary;
var y {k in PROJECTIONS} >= 0, <= U;
var z {j in VOXELS} >= 0;
var z_plus {j in VOXELS} >= 0;
var z_minus {j in VOXELS} >= 0;

# Objective
minimize ObjectiveFunction: sum {j in VOXELS} (quadHelperUnder[j] * z_minus[j] * z_minus[j] + quadHelperOver[j] * z_plus[j] * z_plus[j]);

# Constraints
subject to doses_to_j {j in VOXELS}: z[j] = sum{thisloop in 0..(numloops-1)} (sum{ (n,k,j) in KNJPARAMETERS}( D[n,k,j] * betasparam[n,k] * y[k + 178 * thisloop]));
subject to positive_only {j in VOXELS}: z_plus[j] - z_minus[j] = z[j] - thethreshold[j];

# -------------------------------------------------------------------
## The PP       
# -------------------------------------------------------------------

subject to doses_to_j_yparam {j in VOXELS}: z[j] = sum{thisloop in 0..(numloops-1)} sum{ (n,k,j) in KNJPARAMETERS}( D[n,k,j] * betas[n,k] * yparam[k + 178 * thisloop]);
subject to Mlimits {n in LEAVES}: sum{k in PROJECTIONSM1} mu[n,k] <= Mbar;
subject to abs_greater {n in LEAVES, k in PROJECTIONSM1}: mu[n,k] >= betas[n, k+1] - betas[n,k];
subject to abs_smaller {n in LEAVES, k in PROJECTIONSM1}: mu[n,k] >= -(betas[n, k+1] - betas[n,k]);

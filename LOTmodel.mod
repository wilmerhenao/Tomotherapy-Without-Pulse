# -------------------------------------------------------------------
## The CMP
# -------------------------------------------------------------------
# Set definitions
param numProjections > 0;
param U > 0;
param numLeaves integer > 0;
param projecs > 0;

# Set definitions
set PROJECTIONS = {0..numProjections - 1};
set LEAVES = {0..(numLeaves - 1)};
set VOXELS;
set KNJMPARAMETERS within {n in LEAVES, k in PROJECTIONS, j in VOXELS};

# Parameters
param D {KNJMPARAMETERS} >= 0;
param thethreshold {VOXELS} >= 0;
param quadHelperOver {VOXELS} >= 0;
param quadHelperUnder {VOXELS} >= 0;
param yparam;
#50 msecs is 0.05;
param tprime := 0.04;
param t2prime := 0.15;

# Variables
var betas {n in LEAVES, k in PROJECTIONS} binary;
var z {j in VOXELS} >= 0;
var t {n in LEAVES, k in PROJECTIONS} >= 0;
var z_plus {j in VOXELS} >= 0;
var z_minus {j in VOXELS} >= 0;

# Objective
minimize ObjectiveFunction: sum {j in VOXELS} (quadHelperUnder[j] * z_minus[j] * z_minus[j] + quadHelperOver[j] * z_plus[j] * z_plus[j]);
# -------------------------------------------------------------------
subject to positive_only {j in VOXELS}: z_plus[j] - z_minus[j] = z[j] - thethreshold[j];
subject to doses_to_j_yparam {j in VOXELS}: z[j] = yparam * sum{ (n,k,j) in KNJMPARAMETERS}( D[n,k,j] * betas[n,k]);
subject to upperbound {n in LEAVES, k in PROJECTIONS}: t[n,k] <= ((360/projecs)/(360/60)) * betas[n,k];
subject to lowerbound {n in LEAVES, k in PROJECTIONS}: t[n,k] >= tprime * betas[n,k];
subject to lowerboundaverage: sum{n in LEAVES, k in PROJECTIONS} t[n,k] >= t2prime * sum{n in LEAVES, k in PROJECTIONS} betas[n,k];

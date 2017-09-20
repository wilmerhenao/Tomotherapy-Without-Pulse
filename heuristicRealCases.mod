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
param numLeaves integer > 0;
param numLoops integer > 0;

set PROJECTIONS = {0..numProjections - 1};
set PROJECTIONSM1 = {0..numProjections - 2};
set LEAVES = {0..(numLeaves - 1)};
set LOOPS = {0..numLoops};
set VOXELS;
set KNJMPARAMETERS within {n in LEAVES, k in PROJECTIONS, m in LOOPS, j in VOXELS};

# Parameters
param D {KNJMPARAMETERS} >= 0;
param Mbar = numProjections / 3;
param Loopbar = 20;
param thethreshold {VOXELS} >= 0;
param quadHelperOver {VOXELS} >= 0;
param quadHelperUnder {VOXELS} >= 0;
param betasparam{n in LEAVES, k in PROJECTIONS, m in LOOPS} binary;
param yparam{k in PROJECTIONS} >= 0, <= U;
param yBar{k in PROJECTIONS} >= 0, <= U;

# Variables
var betas {n in LEAVES, k in PROJECTIONS, m in LOOPS} binary;
var mu{n in LEAVES, k in PROJECTIONSM1} binary;
var xi{n in LEAVES, k in PROJECTIONSM1, m in LOOPS} binary;
var y {k in PROJECTIONS} >= 0, <= U;
var z {j in VOXELS} >= 0;
var z_plus {j in VOXELS} >= 0;
var z_minus {j in VOXELS} >= 0;

# Objective
minimize ObjectiveFunction: sum {j in VOXELS} (quadHelperUnder[j] * z_minus[j] * z_minus[j] + quadHelperOver[j] * z_plus[j] * z_plus[j]);

# Constraints
subject to doses_to_j {j in VOXELS}: z[j] = sum{ (n,k,m,j) in KNJMPARAMETERS}( D[n,k,m,j] * betasparam[n,k,m] * y[k]);
subject to positive_only {j in VOXELS}: z_plus[j] - z_minus[j] = z[j] - thethreshold[j];

# -------------------------------------------------------------------
## The DP       
# -------------------------------------------------------------------

subject to doses_to_j_yparam {j in VOXELS}: z[j] = sum{ (n,k,m,j) in KNJMPARAMETERS}( D[n,k,m,j] * betas[n,k,m] * yparam[k]);
subject to Mlimits {n in LEAVES}: sum{k in PROJECTIONSM1} mu[n,k] <= Mbar;
subject to abs_greater {n in LEAVES, k in PROJECTIONSM1}: mu[n,k] >= sum{m in LOOPS} (betas[n, k+1, m] - betas[n,k, m]);
subject to abs_smaller {n in LEAVES, k in PROJECTIONSM1}: mu[n,k] >= sum{m in LOOPS} -(betas[n, k+1, m] - betas[n,k,m]);

# per loop constraints:

subject to Mlimits_perLoop{n in LEAVES, m in LOOPS}: sum{k in PROJECTIONSM1} xi[n,k,m] <= Loopbar;
subject to abs_greater_perloop {n in LEAVES, k in PROJECTIONSM1, m in LOOPS}: xi[n,k,m] >= betas[n, k+1, m] - betas[n,k, m];
subject to abs_smaller_perloop {n in LEAVES, k in PROJECTIONSM1, m in LOOPS}: xi[n,k,m] >= betas[n, k+1, m] - betas[n,k, m];

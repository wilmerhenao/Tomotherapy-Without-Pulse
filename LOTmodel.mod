# -------------------------------------------------------------------
## SOLVE51
# -------------------------------------------------------------------
        # Param definitions
param numProjections > 0;
param U > 0;
param numLeaves integer > 0;
param projecs > 0;
param ko integer > 0;
param kc integer > 0;
param numvoxels > 0;
param r > 0;
param timeperprojection > 0;
param maxkcko := max(ko, kc);
param timeopen >= 0;
param timeclose >= 0;

        # Set definitions
set PROJECTIONS = {-maxkcko..((numProjections - 1) + maxkcko)};
set PROJECTIONSSHORT = {0..(numProjections - 1)};
set PROJECTIONSSHORTM1 = {0..(numProjections - 2)};
set LOTSET = {0..(kc-1)};
set LCTSET = {0..(ko-1)};
set LEAVES = {0..(numLeaves - 1)};
set VOXELS;
set KNJMPARAMETERS within {l in LEAVES, p in PROJECTIONS, j in VOXELS};

# Define indexed collections of sets
set TIMES {LEAVES} default setof {i in -maxkcko..-1} (timeperprojection * i);
set PROJIRREG {LEAVES} ordered default {-maxkcko..0};
set PROJIRREGACTIVE {LEAVES} ordered default {};
set NEWKNJ within {l in LEAVES, p in PROJIRREG[l], v in VOXELS} default {};
set PMAPPER {l in LEAVES, p in PROJECTIONS} default {-maxkcko..0};
#param PMAPPER {PMAPPERSET};
param NEWD {NEWKNJ};
set NEWGRID {l in LEAVES, p in PROJIRREG[l]};
set LOTSETFINE {NEWGRID} default {};
set LCTSETFINE {NEWGRID} default {};

var pnegativesflag; #This var checks if the projections are in negative time
var pcounter; # Number of projections away from zero
var prevtime;

# Parameters
param D {KNJMPARAMETERS} >= 0;
param thethreshold {VOXELS} >= 0;
param quadHelperOver {VOXELS} >= 0;
param quadHelperUnder {VOXELS} >= 0;
param yparam;
#50 msecs is 0.05;

# Variables
var z {j in VOXELS} >= 0;
var z_plus {j in VOXELS} >= 0;
var z_minus {j in VOXELS} >= 0;
var betas {l in LEAVES, p in PROJECTIONS} binary;
var B {l in LEAVES, p in PROJECTIONS} binary;
var cgamma{l in LEAVES, p in PROJECTIONS} binary;
var lgamma{l in LEAVES, p in PROJECTIONS} binary;

var betasFine{l in LEAVES, p in PROJIRREG[l]} binary
var BFine {l in LEAVES, p in PROJIRREG[l]} binary;
var cgammaFine {l in LEAVES, p in PROJIRREG[l]} binary;
var lgammaFine {l in LEAVES, p in PROJIRREG[l]} binary;

# Objective SOLVE51
minimize ObjectiveFunction: sum {j in VOXELS} (quadHelperUnder[j] * z_minus[j] * z_minus[j] + quadHelperOver[j] * z_plus[j] * z_plus[j]);
# -------------------------------------------------------------------
subject to positive_only {j in VOXELS}: z_plus[j] - z_minus[j] = z[j] - thethreshold[j];
subject to doses_to_j_yparam {j in VOXELS}: z[j] = yparam * sum{ (l,p,j) in KNJMPARAMETERS}( D[l,p,j] * betas[l,p]);
subject to LOC {l in LEAVES, p in PROJECTIONSSHORT, k in LOTSET}: B[l,p] <= betas[l, p + k];
subject to LCT {l in LEAVES, p in PROJECTIONSSHORT, k in LCTSET}: cgamma[l,p] <= lgamma[l, p + k];
subject to endOpen {l in LEAVES, p in PROJECTIONSSHORTM1}: betas[l, p] <= betas[l, p + 1] + cgamma[l, p + 1];
subject to endClose {l in LEAVES, p in PROJECTIONSSHORTM1}: lgamma[l, p] <= lgamma[l, p + 1] + B[l, p + 1];
subject to eitherOpenOrClose {l in LEAVES, p in PROJECTIONS}: betas[l, p] + lgamma[l, p] = 1;

# Second PROBLEM
subject to doses_Fine {j in VOXELS}: z[j] = yparam * sum{ (l,p,j) in NEWKNJ}( NEWD[l,p,j] * betasFine[l,p]);
subject to LOC_Fine {l in LEAVES, p in {PROJIRREGACTIVE[l] diff last(PROJIRREGACTIVE[l])} , k in LOTSETFINE[l, p]}: BFine[l,p] <= betasFine[l, p + k];
subject to LCT_Fine {l in LEAVES, p in {PROJIRREGACTIVE[l] diff last(PROJIRREGACTIVE[l])}, k in LCTSETFINE[l, p]}: cgammaFine[l,p] <= lgammaFine[l, p + k];
subject to endOpen_Fine {l in LEAVES, p in {PROJIRREGACTIVE[l] diff last(PROJIRREGACTIVE[l])}}: betasFine[l, p] <= betasFine[l, p + 1] + cgammaFine[l, p + 1];
subject to endClose_Fine {l in LEAVES, p in {PROJIRREGACTIVE[l] diff last(PROJIRREGACTIVE[l])}}: lgammaFine[l, p] <= lgammaFine[l, p + 1] + BFine[l, p + 1];
subject to eitherOpenOrClose_Fine {l in LEAVES, p in PROJIRREG[l]}: betasFine[l, p] + lgammaFine[l, p] = 1;

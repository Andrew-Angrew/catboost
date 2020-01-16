
#include "fold.h"

#include <catboost/private/libs/options/enums.h>

#include <catboost/libs/helpers/restorable_rng.h>

TVector<int> SelectTreesToDrop(
    const double pDrop,
    const int treeCount,
    const EDropoutType dropoutType,
    TRestorableFastRng64* generator
);

/*
void DropTrees(
    const TVector<int>& treesToDrop,
    bool isExpApprox,
    bool dartDownWeight,
    bool doDrop,
    TFold::TBodyTail* bodyTail
);
*/

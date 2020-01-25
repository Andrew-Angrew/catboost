
#include "fold.h"

#include "split.h"

#include <catboost/private/libs/options/enums.h>

#include <catboost/libs/helpers/restorable_rng.h>

TVector<int> SelectTreesToDrop(
    const double pDrop,
    const int treeCount,
    const EDropoutType dropoutType,
    TRestorableFastRng64* generator
);

void DropTreesForSptitSearch(
    const NCB::TTrainingForCPUDataProviders& data,
    const TVector<TSplitTree>& treeStructures,
    const TVector<int>& treesToDrop,
    const bool isExpApprox,
    const bool doDartDownWeighting,
    const double learningRate,
    TFold* fold,
    NPar::TLocalExecutor* localExecutor
);


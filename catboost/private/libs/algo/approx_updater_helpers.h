#pragma once

#include "fold.h"

#include <catboost/private/libs/algo_helpers/approx_updater_helpers.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


struct TLearnProgress;

void ApplyLeafDelta(
    bool storeExpApprox,
    TConstArrayRef<TIndexType> indices,
    TMaybe<TConstArrayRef<ui32>> learnPermutation,
    const TVector<TVector<double>>& treeDelta,
    TVector<TVector<double>>* approx,
    NPar::TLocalExecutor* localExecutor
);

template <bool StoreExpApprox>
inline void UpdateBodyTailApprox(
    const TVector<TVector<TVector<double>>>& treeDelta, // deltas to approxes or to leaf values
    double treeScale,
    TMaybe<TConstArrayRef<TIndexType>> leafIndexes, // if present than treDelta is a diff to leaf values
    NPar::TLocalExecutor* localExecutor,
    TFold* fold
) {
    if (leafIndexes) { // that case should occur only if tree dropout is enabled
        Y_ASSERT(!StoreExpApprox);
        for (int bodyTailId = 0; bodyTailId < fold->BodyTailArr.ysize(); ++bodyTailId) {
            TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
            Y_ASSERT(bt.ApproxBackup.Defined());
            bt.LeafValues.push_back(ScaleElementwise(treeScale, treeDelta[bodyTailId]));
            ApplyLeafDelta(
                /* storeExpApprox */ false,
                *leafIndexes,
                /* learnPermutation */ Nothing(),
                bt.LeafValues.back(),
                bt.ApproxBackup.Get(),
                localExecutor
            );
        }
    } else {
        const auto applyLearningRate = [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
            approx[idx] = UpdateApprox<StoreExpApprox>(
                approx[idx],
                ApplyLearningRate<StoreExpApprox>(delta[idx], treeScale)
            );
        };
        for (int bodyTailId = 0; bodyTailId < fold->BodyTailArr.ysize(); ++bodyTailId) {
            TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
            UpdateApprox(applyLearningRate, treeDelta[bodyTailId], &bt.Approx, localExecutor);
        }
    }
}

void UpdateAvrgApprox(
    bool storeExpApprox,
    bool isDropout,
    ui32 learnSampleCount,
    const TVector<TIndexType>& indices,
    const TVector<TVector<double>>& treeDelta,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData, // can be empty
    TLearnProgress* learnProgress,
    NPar::TLocalExecutor* localExecutor
);

template <class T>
void InitApproxFromBaseline(
    TConstArrayRef<TConstArrayRef<T>> baseline,
    TConstArrayRef<ui32> learnPermutation,
    bool storeExpApproxes,
    bool isDropout,
    TFold::TBodyTail* bodyTail 
) {
    const ui32 learnSampleCount = learnPermutation.size();
    const int approxDimension = bodyTail->Approx.ysize();
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (ui32 docId : xrange(0u, static_cast<ui32>(bodyTail->TailFinish))) {
            ui32 initialIdx = docId;
            if (docId < learnSampleCount) {
                initialIdx = learnPermutation[docId];
            }
            bodyTail->Approx[dim][docId] = baseline[dim][initialIdx];
        }
    }
    if (isDropout) {
        bodyTail->ApproxBackup = bodyTail->Approx;
    }
    ExpApproxIf(storeExpApproxes, &bodyTail->Approx);
}


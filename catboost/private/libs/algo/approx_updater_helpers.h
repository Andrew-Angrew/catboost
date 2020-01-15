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

template <bool StoreExpApprox>
inline void UpdateBodyTailApprox(
    const TVector<TVector<TVector<double>>>& approxDelta,
    double learningRate,
    NPar::TLocalExecutor* localExecutor,
    TFold* fold
) {
    const auto applyLearningRate = [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
        approx[idx] = UpdateApprox<StoreExpApprox>(
            approx[idx],
            ApplyLearningRate<StoreExpApprox>(delta[idx], learningRate)
        );
    };
    for (int bodyTailId = 0; bodyTailId < fold->BodyTailArr.ysize(); ++bodyTailId) {
        TFold::TBodyTail& bt = fold->BodyTailArr[bodyTailId];
        UpdateApprox(applyLearningRate, approxDelta[bodyTailId], &bt.Approx, localExecutor);
    }
}

void UpdateAvrgApprox(
    bool storeExpApprox,
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
        bodyTail->AllTreeApprox = bodyTail->Approx;
    }
    ExpApproxIf(storeExpApproxes, &bodyTail->Approx);
}


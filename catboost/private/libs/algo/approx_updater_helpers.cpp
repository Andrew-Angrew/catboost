#include "approx_updater_helpers.h"

#include "learn_context.h"

#include <catboost/private/libs/functools/forward_as_const.h>

#include <util/generic/cast.h>


using namespace NCB;


template <bool StoreExpApprox, bool UseLearnPermutation>
static void ApplyLeafDelta(
    TConstArrayRef<TIndexType> indices,
    TConstArrayRef<ui32> learnPermutation,
    const TVector<TVector<double>>& treeDelta,
    TVector<TVector<double>>* approx,
    NPar::TLocalExecutor* localExecutor
) {
    const auto updateApprox = [indices, learnPermutation] (
        TConstArrayRef<double> delta,
        TArrayRef<double> approx,
        size_t idx
    ) {
        if constexpr (UseLearnPermutation) {
            approx[learnPermutation[idx]] = UpdateApprox<StoreExpApprox>(approx[idx], delta[indices[idx]]);
        } else {
            Y_UNUSED(learnPermutation);
            approx[idx] = UpdateApprox<StoreExpApprox>(approx[idx], delta[indices[idx]]);
        }
    };
    TVector<TVector<double>> expTreeDelta;
    if constexpr (StoreExpApprox) {
        expTreeDelta = treeDelta;
        ExpApproxIf(StoreExpApprox, &expTreeDelta);
    }
    const auto* properTreeDelta = StoreExpApprox ? &expTreeDelta : &treeDelta;
    UpdateApprox(updateApprox, *properTreeDelta, approx, localExecutor);
}

void ApplyLeafDelta(
    bool storeExpApprox,
    TConstArrayRef<TIndexType> indices,
    TMaybe<TConstArrayRef<ui32>> learnPermutation,
    const TVector<TVector<double>>& treeDelta,
    TVector<TVector<double>>* approx,
    NPar::TLocalExecutor* localExecutor
) {
    ForwardArgsAsIntegralConst(
        [&] (auto storeExpApproxAsConst, auto useLearnPermutation) {
            ::ApplyLeafDelta<storeExpApproxAsConst, useLearnPermutation>(
                indices,
                *learnPermutation,
                treeDelta,
                approx,
                localExecutor
            );
        },
        storeExpApprox, learnPermutation.Defined()
    );
}

void UpdateAvrgApprox(
    bool storeExpApprox,
    bool isDropout,
    ui32 learnSampleCount,
    const TVector<TIndexType>& indices,
    const TVector<TVector<double>>& treeDelta,
    TConstArrayRef<TTrainingForCPUDataProviderPtr> testData, // can be empty
    TLearnProgress* learnProgress,
    NPar::TLocalExecutor* localExecutor
) {
    Y_ASSERT(learnProgress->AveragingFold.BodyTailArr.ysize() == 1);
    const TVector<size_t>& testOffsets = CalcTestOffsets(learnSampleCount, testData);

    localExecutor->ExecRange(
        [&](int setIdx){
            if (setIdx == 0) { // learn data set
                if (learnSampleCount == 0) {
                    return;
                }
                TFold::TBodyTail& bt = learnProgress->AveragingFold.BodyTailArr[0];
                auto* approx = &bt.Approx;
                if (isDropout) {
                    Y_ASSERT(bt.ApproxBackup.Defined());
                    approx = bt.ApproxBackup.Get();
                }
                Y_ASSERT(bt.Approx[0].ysize() == bt.TailFinish);
                TConstArrayRef<TIndexType> indicesRef(indices);
                ApplyLeafDelta(
                    storeExpApprox && !isDropout,
                    indicesRef,
                    /* learnPermutation */ Nothing(),
                    treeDelta,
                    approx,
                    localExecutor
                );

                TConstArrayRef<ui32> learnPermutationRef(learnProgress->AveragingFold.GetLearnPermutationArray());
                Y_ASSERT(learnProgress->AvrgApprox[0].size() == learnSampleCount);
                ApplyLeafDelta(
                    /* storeExpApprox */ false,
                    indicesRef, 
                    learnPermutationRef,
                    treeDelta,
                    &learnProgress->AvrgApprox,
                    localExecutor
                );
            } else { // test data set
                const int testIdx = setIdx - 1;
                const size_t testSampleCount = testData[testIdx]->GetObjectCount();
                TConstArrayRef<TIndexType> indicesRef(indices.data() + testOffsets[testIdx], testSampleCount);
                const auto updateTestApprox = [=](
                    TConstArrayRef<double> delta,
                    TArrayRef<double> approx,
                    size_t idx
                ) {
                    approx[idx] += delta[indicesRef[idx]];
                };
                Y_ASSERT(learnProgress->TestApprox[testIdx][0].size() == testSampleCount);
                UpdateApprox(updateTestApprox, treeDelta, &learnProgress->TestApprox[testIdx], localExecutor);
            }
        },
        0,
        1 + SafeIntegerCast<int>(testData.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

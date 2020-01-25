#include "tree_dropout.h"

#include "approx_updater_helpers.h"
#include "index_calcer.h"

TVector<int> SelectTreesToDrop(
    const double pDrop,
    const int treeCount,
    const EDropoutType dropoutType,
    TRestorableFastRng64* generator
) {
    const auto treeScores = GenRandUniformVector(treeCount, generator->GenRand());
    TVector<int> treesToDrop;
    for (int treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
        if (treeScores[treeIndex] < pDrop) {
            treesToDrop.push_back(treeIndex);
        }
    }
    if (dropoutType == EDropoutType::Dart && treeCount > 0 && treesToDrop.empty()) {
        treesToDrop.push_back(static_cast<int>(generator->Uniform(treeCount)));
    }
    return treesToDrop;
}

static void DropTree(
    const TVector<TIndexType>& leafIndices,
    const int treeIndex,
    const double dartDownWeighting,
    TFold* fold,
    NPar::TLocalExecutor* localExecutor
) {
    NPar::ParallelFor(
        *localExecutor,
        0,
        fold->BodyTailArr.size(),
        [=, &leafIndices] (ui32 bodyTailId) {
            auto& bodyTail = fold->BodyTailArr[bodyTailId];
            const auto& leafValues = bodyTail.LeafValues[treeIndex];
            ApplyLeafDelta(
                /* storeExpApprox */ false,
                leafIndices,
                /* learnPermutation */ Nothing(),
                ScaleElementwise(-1, leafValues),
                &bodyTail.Approx,
                localExecutor
            );
            if (dartDownWeighting != 0.0) { // TODO(strashila): reuse approx delta in that case
                ApplyLeafDelta(
                    /* storeExpApprox */ false,
                    leafIndices,
                    /* learnPermutation */ Nothing(),
                    ScaleElementwise(-dartDownWeighting, leafValues),
                    bodyTail.ApproxBackup.Get(),
                    localExecutor
                );
                bodyTail.LeafValues[treeIndex] = ScaleElementwise(1 - dartDownWeighting, leafValues);
            }
        }
    );
}

void DropTreesForSptitSearch(
    const NCB::TTrainingForCPUDataProviders& data,
    const TVector<TSplitTree>& treeStructures,
    const TVector<int>& treesToDrop,
    const bool isExpApprox,
    const bool doDartDownWeighting,
    const double learningRate,
    TFold* fold,
    NPar::TLocalExecutor* localExecutor
) {
    NPar::ParallelFor(
        *localExecutor,
        0,
        fold->BodyTailArr.size(),
        [fold, localExecutor] (ui32 bodyTailId) {
            auto* bodyTail = &fold->BodyTailArr[bodyTailId];
            CopyApprox(*bodyTail->ApproxBackup, &bodyTail->Approx, localExecutor);
        }
    );
    if (!treesToDrop.empty()) {
        const double dartDownWeighting = (
            doDartDownWeighting
            ? learningRate / (static_cast<double>(treesToDrop.size()) + learningRate)
            : 0.0
        );
        for (int treeIndex : treesToDrop) {
            const auto leafIndices = BuildIndices(
                *fold,
                treeStructures[treeIndex],
                data.Learn,
                /* testData */ {},
                localExecutor
            );
            DropTree(leafIndices, treeIndex, dartDownWeighting, fold, localExecutor);
        }
    }

    NPar::ParallelFor(
        *localExecutor,
        0,
        fold->BodyTailArr.size(),
        [fold, isExpApprox] (ui32 bodyTailId) {
            ExpApproxIf(isExpApprox, &fold->BodyTailArr[bodyTailId].Approx); // TODO(strashila): parallelize
        }
    );
}


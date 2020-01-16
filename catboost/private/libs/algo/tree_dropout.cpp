#include "tree_dropout.h"


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

/*
void DropTrees(
    const TVector<int>& treesToDrop,
    bool isExpApprox,
    bool dartDownWeight,
    bool doDrop,
    TFold::TBodyTail* bodyTail
) {
    
}
*/

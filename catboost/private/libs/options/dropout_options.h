#pragma once

#include "enums.h"
#include "option.h"

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    class TTreeDropoutOptions {
    public:
        TTreeDropoutOptions();

        bool operator==(const TTreeDropoutOptions& rhs) const;
        bool operator!=(const TTreeDropoutOptions& rhs) const;

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        void Validate() const;

    public:
        TOption<EDropoutType> DropoutType;
        TOption<float> TreeDropProbability;
        TOption<bool> ForSplitSearchOnly;
    };
}

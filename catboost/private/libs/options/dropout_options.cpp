#include "dropout_options.h"
#include "json_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <library/json/json_value.h>


NCatboostOptions::TTreeDropoutOptions::TTreeDropoutOptions()
    : DropoutType("dropout_type", EDropoutType::None)
    , TreeDropProbability("p_drop", 0.0f)
    , ForSplitSearchOnly("drop_trees_for_split_search_only", false)
{}

bool NCatboostOptions::TTreeDropoutOptions::operator==(const TTreeDropoutOptions& rhs) const {
    return std::tie(DropoutType, TreeDropProbability, ForSplitSearchOnly) ==
       std::tie(rhs.DropoutType, rhs.TreeDropProbability, rhs.ForSplitSearchOnly);
}

bool NCatboostOptions::TTreeDropoutOptions::operator!=(const TTreeDropoutOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TTreeDropoutOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &DropoutType, &TreeDropProbability, &ForSplitSearchOnly);
    Validate();
}

void NCatboostOptions::TTreeDropoutOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, DropoutType, TreeDropProbability, ForSplitSearchOnly);
}

void NCatboostOptions::TTreeDropoutOptions::Validate() const {
    CB_ENSURE(TreeDropProbability.Get() >= 0.0f, "tree dropout probability should be non negative.");
    CB_ENSURE(TreeDropProbability.Get() < 1.0f, "tree dropout probability should be strictly less than 1.");
}

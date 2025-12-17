#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "config.hpp"

namespace MNN {
namespace Express {
class OnnxGatherTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int axis    = 0;
        auto op     = expr->get();
        if (nullptr != op->main_as_Extra()->attr()) {
            for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                    break;
                }
            }
        }
        auto axisVar = _Scalar<int>(axis);
        auto config = Global<modelConfig>::Get();
        if (config->optimizeLevel < 2) {
            // Add negative protect, may decrease performance
            auto rankVar = _Rank(inputs[0]);
            axisVar = _Mod(axisVar + rankVar, rankVar);
            auto shapeVar = _Shape(inputs[0], true);
            auto axisLengthVar = _Squeeze(_StridedSlice(shapeVar, _Unsqueeze(axisVar, {0}), _Unsqueeze(axisVar + _Scalar<int>(1), {0}),  _Unsqueeze(_Scalar<int32_t>(1), {0}), 0, 0, 0, 0, 0));
            inputs[1] = _Mod(inputs[1] + axisLengthVar, axisLengthVar);
        }
        auto output = _GatherV2(inputs[0], inputs[1], axisVar);
        output->setName(expr->name());
        return output->expr().first;
    }
};
static auto gRegister = []() {
    OnnxExtraManager::get()->insert("Gather", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherTransform));
    return true;
}();

} // namespace Express
} // namespace MNN

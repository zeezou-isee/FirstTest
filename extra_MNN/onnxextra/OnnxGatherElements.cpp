#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "config.hpp"

namespace MNN {
namespace Express {
class OnnxGatherElementTransform : public OnnxExtraManager::Transform {
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
        if (inputs.size() < 2) {
            MNN_ERROR("GatherElements should has two inputs\n");
            return nullptr;
        }
        // Reshape the input as outside, axis, inside
        auto index = inputs[1];
        auto input = inputs[0];
        auto dst = Express::_GatherElements(input, index, _Scalar(axis));
        dst->setName(expr->name());
        return dst->expr().first;
    }
};
static auto gRegister = []() {
    OnnxExtraManager::get()->insert("GatherElements", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherElementTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
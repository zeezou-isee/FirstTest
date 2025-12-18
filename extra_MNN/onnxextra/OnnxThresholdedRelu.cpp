#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxThresholdedReluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        float alpha = 1;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "alpha") {
                    alpha = attr->f();
                }
            }
        }
        auto input = expr->inputs()[0];
        auto res = _Select(_Greater(input, _Const(alpha)), input, _Const(0.0f));
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};
static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("ThresholdedRelu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxThresholdedReluTransform));
    
    return true;
}();

} // namespace Express
} // namespace MNN

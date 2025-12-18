#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxShrinkTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        float bias = 0, lambd = 0.5;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "bias") {
                    bias = attr->f();
                } else if (attr->key()->str() == "lambd") {
                    lambd = attr->f();
                }
            }
        }
        auto input = expr->inputs()[0];
        auto biasVar = _Const(bias);
        auto res = _Select(_Greater(input, _Const(lambd)), _Subtract(input, biasVar), // x-bias for x > lambd
                        _Select(_Less(input, _Const(-lambd)), _Add(input, biasVar), // x+bias for x < -lambd
                            _Const(0.0))); // 0 for otherwise
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("Shrink", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxShrinkTransform));
  
    return true;
}();

} // namespace Express
} // namespace MNN
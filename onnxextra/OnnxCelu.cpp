#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxCeluTransform : public OnnxExtraManager::Transform {
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
        auto alphaVar = _Const(alpha);
        auto y = _Multiply(_Subtract(_Exp(_Divide(input, alphaVar)), _Const(1.0f)), alphaVar);
        auto res = _Select(_Less(input, _Const(0.0f)), y, input);
        auto newExpr = res->expr().first;
        newExpr->setName(expr->name());
        return newExpr;
    }
};
static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("Celu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxCeluTransform));
    
    return true;
}();

} // namespace Express
} // namespace MNN
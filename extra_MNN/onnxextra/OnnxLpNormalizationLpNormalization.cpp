#include <math.h>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxLpNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto input = expr->inputs()[0];
        int p = 2, axis = -1;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                auto attrName = attr->key()->str();
                if (attrName == "axis") {
                    axis = attr->i();
                } else if (attrName == "p") {
                    p = attr->i();
                }
            }
        }
        if (p != 1 && p != 2) {
            MNN_ERROR("Onnx's LpNormalization only support attr p is 1 or 2");
            return nullptr;
        }
        VARP res;
        if (p == 1) {
            res = input / _ReduceSumMutable(_Abs(input), _Scalar<int>(axis), true);
        } else {
            res = input * _Rsqrt(_ReduceSumMutable(input * input, _Scalar<int>(axis), true));
        }
        res->setName(expr->name());
        return res->expr().first;
    }
};

static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("LpNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLpNormTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
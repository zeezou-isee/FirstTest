#include <math.h>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxMeanVarianceNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        std::vector<int> axes {0, 2, 3};
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "axes") {
                    axes.clear();
                    for (int i = 0; i < attr->list()->i()->size(); ++i) {
                        axes.push_back(attr->list()->i()->Get(i));
                    }
                }
            }
        }
        auto input = expr->inputs()[0];
        auto mean = _ReduceMean(input, axes, true);
        auto temp = input - mean;
        auto var = _ReduceMean(temp * temp, axes, true);
        auto res = temp * _Rsqrt(var);
        res->setName(expr->name());
        return res->expr().first;
    }
};
static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("MeanVarianceNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxMeanVarianceNormTransform));
   
    return true;
}();

} // namespace Express
} // namespace MNN
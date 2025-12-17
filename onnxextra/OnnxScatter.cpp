#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxScatterElementsTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs   = expr->inputs();
        int axis      = 0;
        auto op       = expr->get();
        int reduction = -1;
        if (nullptr != op->main_as_Extra()->attr()) {
            for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                    break;
                }
                if (key == "reduction") {
                    auto reductionStr = attr->s()->str();
                    if (reductionStr == "add") {
                        reduction = BinaryOpOperation_ADD;
                    } else if (reductionStr == "mul") {
                        reduction = BinaryOpOperation_MUL;
                    }
                    break;
                }
            }
        }
        auto input = inputs[0], indice = inputs[1], update = inputs[2];
        auto dst   = Express::_ScatterElements(input, indice, update, _Scalar(axis), reduction);
        dst->setName(expr->name());
        return dst->expr().first;
    }
};

static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("Scatter",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxScatterElementsTransformer));
   
    return true;
}();

} // namespace Express
} // namespace MNN

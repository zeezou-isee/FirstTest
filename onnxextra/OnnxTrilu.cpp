#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxTriluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto shape = _Shape(inputs[0]), zero = _Scalar<int>(0), oneV = _Unsqueeze(_Scalar<int>(1), {0});
        auto H = _Slice(shape, _Unsqueeze(_Scalar<int>(-2), {0}), oneV), W = _Slice(shape, _Unsqueeze(_Scalar<int>(-1), {0}), oneV);
        auto rangeH = _Unsqueeze(_Range(zero, H, oneV), {1}), rangeW = _Unsqueeze(_Range(zero, W, oneV), {0});
        bool upper = true;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                if (attr->key()->str() == "upper") {
                    upper = attr->i();
                }
            }
        }
        auto k = (inputs.size() == 2 ? inputs[1] : _Scalar<int>(0));
        auto mask = (upper ? _GreaterEqual(rangeW, rangeH + k) : _GreaterEqual(rangeH, rangeW - k));
        mask = _Reshape(mask, _Concat({_Fill(_Unsqueeze(_Size(shape) - _Scalar<int>(2), {0}), oneV), _Shape(mask)}, 0));
        auto res = _Select(mask, inputs[0], zero);
        res->setName(expr->outputName(0));
        return res->expr().first;
    }
};
static auto gRegister = []() {
   
    OnnxExtraManager::get()->insert("Trilu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxTriluTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
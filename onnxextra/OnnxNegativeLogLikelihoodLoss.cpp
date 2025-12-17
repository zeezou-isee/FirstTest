#include <algorithm>

#include <MNN/expr/Expr.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxSoftmaxCrossEntropyLossTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_THROW_CHECK(inputs.size() == 2 || inputs.size() == 3, "Onnx SoftmaxCrossEntropyLoss needs 2 or 3 inputs.");
        MNN_THROW_CHECK(expr->outputSize() == 1, "MNN SoftmaxCrossEntropyLoss only support 1 output\n");

        int ignore_index = -1;
        std::string reduction = "mean";
        auto attrs = expr->get()->main_as_Extra()->attr();
        for (auto it = attrs->begin(); it != attrs->end(); ++it) {
            if (it->key()->str() == "ignore_index") {
                ignore_index = it->i();
            } else if (it->key()->str() == "reduction") {
                reduction = it->s()->str();
            }
        }
        auto shape = _Shape(inputs[0], true), oneV = _Unsqueeze(_Scalar<int>(1), {0}), classes = _Slice(shape, oneV, oneV);
        auto mask = _OneHot(inputs[1], classes, _Scalar<float>(1), _Scalar<float>(0), 1);
        mask = mask * _Cast<float>(_Unsqueeze(_NotEqual(inputs[1], _Scalar<int>(ignore_index)), {1}));
        
        auto log_prob = inputs[0];
        if (expr->get()->main_as_Extra()->type()->str() == "SoftmaxCrossEntropyLoss") {
            log_prob = _Log(_Softmax(inputs[0], 1));
        }
        auto temp = log_prob;
        VARP weight(nullptr);
        if (inputs.size() == 3) {
            auto weightShape = _Concat({_Unsqueeze(classes, {0}), _Fill(_Size(shape) - _Scalar<int>(2), oneV)}, 0);
            weight = _Reshape(inputs[2], weightShape);
            temp = temp * weight;
        }
        auto output = _ReduceSum(mask * _Negative(temp), {1}, false);
        if (reduction == "sum") {
            output = _ReduceSum(output);
        } else if (reduction == "mean") {
            if (inputs.size() == 3) {
                output = _ReduceSum(output) / _ReduceSum(weight * mask);
            } else {
                output = _ReduceMean(output);
            }
        }

        output->setName(expr->outputName(0));
        return output->expr().first;
    }
};
static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("NegativeLogLikelihoodLoss",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxSoftmaxCrossEntropyLossTransform));
    return true;
}();

} // namespace Express
} // namespace MNN

#include <math.h>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

class OnnxLayerNormTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto input = expr->inputs()[0];
        int axis = -1;
        float eps = 1e-05;
        auto attrs = expr->get()->main_as_Extra()->attr();
        if (attrs != nullptr) {
            for (const auto& attr : *attrs) {
                auto attrName = attr->key()->str();
                if (attrName == "axis") {
                    axis = attr->i();
                }
                if (attrName == "epsilon") {
                    eps = attr->f();
                }
            }
        }
        if (expr->outputSize() > 1 || axis > 0) {
            // If axis > 0, we can't determine how many axis should be norm
            auto axisVar = _Scalar<int>(axis);
            // Add negative protect, may decrease performance
            auto rankVar = _Rank(inputs[0]);
            axisVar = _Mod(axisVar + rankVar, rankVar);
            auto reduceAxis = _Range(axisVar, rankVar, _Scalar<int>(1));
            auto mean = _ReduceMeanMutable(input, reduceAxis, true);
            auto sub = input - mean;
            auto normal = _Rsqrt(_ReduceMeanMutable(_Square(sub), reduceAxis, true) + _Scalar<float>(eps));
            auto y = sub * normal * inputs[1];
            if (inputs.size() > 2) {
                y = y + inputs[2];
            }
            std::vector<VARP> identityOutputs = {y};
            if (expr->outputSize() > 1) {
                identityOutputs.emplace_back(mean);
            }
            if (expr->outputSize() > 2) {
                identityOutputs.emplace_back(normal);
            }
            std::unique_ptr<OpT> copyOp(new OpT);
            copyOp->type = OpType_Identity;
            auto resultExpr = Expr::create(copyOp.get(), identityOutputs, identityOutputs.size());
            resultExpr->setName(expr->name());
            for (int i=0; i<expr->outputSize(); ++i) {
                auto var = MNN::Express::Variable::create(resultExpr, i);
                var->setName(expr->outputName(i));
            }
            return resultExpr;
        }
        std::shared_ptr<MNN::OpT> layernorm(new MNN::OpT);
        layernorm->type = OpType_LayerNorm;
        layernorm->main.value = new LayerNormT;
        layernorm->main.type = OpParameter_LayerNorm;
        auto param = layernorm->main.AsLayerNorm();
        param->axis.resize(-axis);
        for (int i=0; i<param->axis.size(); ++i) {
            param->axis[i] = i-(int)(param->axis.size());
        }
        param->epsilon = eps;
        const float* scalePtr = nullptr;
        const float* biasPtr = nullptr;
        if (inputs.size() > 1) {
            scalePtr = inputs[1]->readMap<float>();
        }
        if (nullptr != scalePtr) {
            param->gamma.resize(inputs[1]->getInfo()->size);
            ::memcpy(param->gamma.data(), scalePtr, param->gamma.size() * sizeof(float));
            param->beta.resize(inputs[1]->getInfo()->size);
            ::memset(param->beta.data(), 0, param->gamma.size() * sizeof(float));
        }
        if (inputs.size() > 2 && nullptr != scalePtr) {
            biasPtr = inputs[2]->readMap<float>();
        }
        if (nullptr != biasPtr) {
            ::memcpy(param->beta.data(), biasPtr, param->gamma.size() * sizeof(float));
        }
        auto layerexpr = Expr::create(layernorm.get(), {input});
        auto output = Variable::create(layerexpr);
        if (scalePtr == nullptr) {
            if (inputs.size() > 1) {
                output = output * inputs[1];
            }
        }
        if (biasPtr == nullptr) {
            if (inputs.size() > 2) {
                output = output + inputs[2];
            }
        }
        output->setName(expr->name());
        return output->expr().first;
    }
};


static auto gRegister = []() {
    OnnxExtraManager::get()->insert("LayerNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLayerNormTransform));
    return true;
}();

} // namespace Express
} // namespace MNN

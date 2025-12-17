#include <math.h>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {

static VARP _ReshapeF(VARP x, VARP shape, MNN::MNN_DATA_FORMAT format) {
    MNN_ASSERT(nullptr != x);
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type                      = OpType_Reshape;
    reshape->main.type                 = OpParameter_Reshape;
    reshape->main.value                = new ReshapeT;
    reshape->main.AsReshape()->dimType = format;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}

static VARP _OnnxReshape(VARP x, VARP shape) {
    std::unique_ptr<OpT> reshape(new OpT);
    reshape->type = OpType_Reshape;
    reshape->main.type = OpParameter_Reshape;
    reshape->main.value = new ReshapeT;
    reshape->main.AsReshape()->dimType = MNN_DATA_FORMAT_NCHW;
    return (Variable::create(Expr::create(reshape.get(), {x, shape})));
}

class OnnxInstanceNormalTransform : public OnnxExtraManager::Transform {
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        MNN_THROW_CHECK(inputs.size() == 3, "InstanceNormal should have 3 inputs");
        auto input = inputs[0];

        int channels  = 1;
        float epsilon = 1e-10;

        auto bnOp       = expr->get();
        auto extraParam = bnOp->main_as_Extra();
        int size        = 0;
        if (nullptr != extraParam->attr()) {
            size = extraParam->attr()->size();
            for (int i = 0; i < size; ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto& key = attr->key()->str();
                if (key == "epsilon") {
                    epsilon = attr->f();
                }
            }
        }
        bool needScale = true;
        bool scaleConst = false;
        do {
            auto biasPtr = inputs[2]->readMap<float>();
            auto scalePtr = inputs[1]->readMap<float>();
            if (nullptr == biasPtr || nullptr == scalePtr) {
                break;
            }
            scaleConst = true;
            auto oneVar = _Scalar<float>(1.0f);
            auto scaleOff = inputs[1] - oneVar;
            auto scaleSum = _ReduceSum(scaleOff * scaleOff);
            if (scaleSum->readMap<float>()[0] > 0.000001f) {
                break;
            }
            auto biasSum = _ReduceSum(inputs[2] * inputs[2]);
            if (biasSum->readMap<float>()[0] > 0.000001f) {
                break;
            }
            needScale = false;
        } while (false);
        auto originShape = _Shape(inputs[0], NCHW);
        auto inputDim3 = _Reshape(inputs[0], {0, 0, -1}, NCHW);
        
        // Turn to layernorm
        std::unique_ptr<MNN::OpT> layerNormOp(new MNN::OpT);
        layerNormOp->type = OpType_LayerNorm;
        layerNormOp->main.value = new LayerNormT;
        layerNormOp->main.type = OpParameter_LayerNorm;
        {
            auto param = layerNormOp->main.AsLayerNorm();
            param->axis = {-1}; // Layernorm only need axis's size as 1
            param->epsilon = epsilon;
            param->group = 1;
        }
        auto res = Variable::create(Expr::create(layerNormOp.get(), {inputDim3}));
        res = _ReshapeF(res, originShape, MNN_DATA_FORMAT_NCHW);
        if (needScale) {
            if (scaleConst) {
                auto biasPtr = inputs[2]->readMap<float>();
                auto scalePtr = inputs[1]->readMap<float>();
                int channels = inputs[1]->getInfo()->size;
                std::vector<float> scales(channels);
                std::vector<float> bias(channels);
                ::memcpy(bias.data(), biasPtr, channels * sizeof(float));
                ::memcpy(scales.data(), scalePtr, channels * sizeof(float));
                res = _Scale(res, channels, std::move(scales), std::move(bias));
            } else {
                auto compatShape = _Concat({_Shape(inputs[1], true), _Fill(_Unsqueeze(_Size(_Shape(input, true)) - _Scalar<int>(2), {0}), _Scalar<int>(1))}, 0);
                auto scale      = _OnnxReshape(inputs[1], compatShape);
                auto bias       = _OnnxReshape(inputs[2], compatShape);
                res = res * scale + bias;
            }
        }
        res->setName(expr->name());
        return res->expr().first;
    }
};


static auto gRegister = []() {
    
    OnnxExtraManager::get()->insert("InstanceNormalization",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxInstanceNormalTransform));
   
    return true;
}();

} // namespace Express
} // namespace MNN
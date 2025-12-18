#include <numeric>

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxScatterNdTransformer : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto op = expr->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto inputs = expr->inputs();
        if (3 != inputs.size()) {
            MNN_ERROR("Onnx ScatterND error for inputs: %d\n", (int)inputs.size());
            return nullptr;
        }
        // Onnx Scatter = data + MNN::Scatter(indice, update, shape)
        auto config = Global<modelConfig>::Get();
        auto data   = inputs[0];
        auto info   = data->getInfo();
        auto type   = halide_type_of<float>();
        if (nullptr != info) {
            type = info->type;
        }
        auto shape  = _Shape(data, true);
        auto indice = inputs[1];
        auto update = inputs[2];
        if (config->optimizeLevel < 2) {
            auto indiceShape = _Shape(indice, true);
            auto indiceRank = _Rank(indice);
            auto lastDim = _Slice(indiceShape, _Unsqueeze(indiceRank - _Scalar<int>(1), {0}), _Unsqueeze(_Scalar<int>(1), {0}));
            auto clipShape = _Slice(shape, _Unsqueeze(_Scalar<int>(0), {0}), lastDim);
            indice = _Mod(indice + clipShape, clipShape);
        }
        auto version = config->targetVersion;
        if (version < 2.0f) {
            // For target version < 2.0 , don't support 4 input scatternd
            auto tfRes  = _ScatterNd(indice, update, shape);
            VARP tfMask;
            if (type.code == halide_type_float) {
                auto updateOne = _Fill(_Shape(update, NCHW), _Scalar<float>(1.0f));
                auto mask = _ScatterNd(indice, updateOne, shape);
                tfMask = _Cast<float>(_Less(mask, _Scalar<float>(0.5f)));
            } else {
                auto updateOne = _Fill(_Shape(update, NCHW), _Scalar<int>(1));
                auto mask = _ScatterNd(indice, updateOne, shape);
                tfMask = _Less(mask, _Scalar<int>(1));
            }
            auto dst    = data * tfMask + tfRes;
            dst->setName(expr->name());
            return dst->expr().first;
        }
        auto tfRes  = _ScatterNd(indice, update, shape, data);
        tfRes->setName(expr->name());
        return tfRes->expr().first;
    }
};
static auto gRegister = []() {
    OnnxExtraManager::get()->insert("ScatterND",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxScatterNdTransformer));
    
    return true;
}();

} // namespace Express
} // namespace MNN
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxPreluTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_THROW_CHECK(inputs.size() == 2, "Onnx Prelu Should have 2 inputs!");

        auto slope     = inputs[1];
        auto slopeInfo = slope->getInfo();
        auto slopeData = slope->readMap<float>();
        if (slopeInfo == nullptr || slopeData == nullptr) {
            auto k = _Select(_Less(inputs[0], _Scalar<float>(0)), slope, _Scalar<float>(1));
            auto res = _Multiply(inputs[0], k);
            res->setName(expr->outputName(0));
            return res->expr().first;
        }

        const int slopeSize = slopeInfo->size;

        std::unique_ptr<PReluT> preluParam(new PReluT);

        preluParam->slopeCount = slopeSize;

        preluParam->slope.resize(slopeSize);
        memcpy(preluParam->slope.data(), slopeData, slopeSize * sizeof(float));

        // prelu(input, slope) => mergedPrelu(input)
        std::unique_ptr<OpT> mergedOp(new OpT);
        mergedOp->name       = expr->name();
        mergedOp->type       = OpType_PReLU;
        mergedOp->main.type  = OpParameter_PRelu;
        mergedOp->main.value = preluParam.release();
        auto newExpr         = Expr::create(mergedOp.get(), {inputs[0]});
        newExpr->setName(expr->name());
        return newExpr;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("PRelu", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxPreluTransform));
   
    return true;
}();

} // namespace Express
} // namespace MNN
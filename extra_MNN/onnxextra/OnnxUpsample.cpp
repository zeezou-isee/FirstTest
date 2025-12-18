#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {
class OnnxUpSampleTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        std::vector<float> scales;
        int scalesSize = 1;

        auto op            = expr->get();
        auto extraParam    = op->main_as_Extra();
        const int attrSize = extraParam->attr()->size();
        std::string interpMode;
        std::string coordMode = ""; // detect align_corner attribute
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "mode") {
                interpMode = attr->s()->str();
            } else if ((inputs.size() == 1) && key == "scales") {
                scalesSize = attr->list()->f()->size();
                scales.resize(scalesSize);
                memcpy(scales.data(), attr->list()->f()->data(), sizeof(float) * scalesSize);
            } else if (key == "coordinate_transformation_mode") {
                coordMode = attr->s()->str();
            }
        }

        std::unique_ptr<OpT> mergeredUpsample(new OpT);
        mergeredUpsample->name      = expr->name();
        mergeredUpsample->type      = OpType_Interp;
        mergeredUpsample->main.type = OpParameter_Interp;

        std::unique_ptr<InterpT> interpParam(new InterpT);

        const float* scaleDataPtr = scales.data();

        if (inputs.size() == 2) {
            auto scale     = inputs[1];
            scaleDataPtr   = scale->readMap<float>();
            auto scaleInfo = scale->getInfo();

            if (!scaleDataPtr) {
                mergeredUpsample->main.value = interpParam.release();
                auto output = Variable::create(Expr::create(mergeredUpsample.get(), {inputs[0], inputs[1]}));
                return output->expr().first;
            }
            // scale is constant node
            scalesSize = scaleInfo->size;
        }

        interpParam->widthScale  = 1.0f;
        interpParam->heightScale = 1.0f;
        if (scalesSize >= 2 && scalesSize <= 4) {
            MNN_THROW_CHECK(scaleDataPtr[1] == 1.0f, "MNN NOT SUPPORT Upsamle along with channle");
            if (scalesSize >= 3) {
                interpParam->heightScale = scaleDataPtr[2];
            }
            if (scalesSize == 4){
                interpParam->widthScale  = scaleDataPtr[3];
            } 
        } else {
            MNN_ERROR("MNN Not support Upsample when scale size = %d\n", scalesSize);
        }
        interpParam->alignCorners = (coordMode == "align_corners");

        // 1:near 2: bilinear 3: cubic
        if (interpMode == "nearest") {
            interpParam->resizeType = 1;
        } else if (interpMode == "bilinear" || interpMode == "linear") {
            interpParam->resizeType = 2;
        } else if (interpMode == "cubic") {
            interpParam->resizeType = 3;
        } else {
            MNN_ERROR("Unsupported Upsample mode! ==> %s\n", interpMode.c_str());
        }

        mergeredUpsample->main.value = interpParam.release();
        auto newInput                = inputs[0];
        auto tempOutput              = Variable::create(Expr::create(mergeredUpsample.get(), {newInput}));
        tempOutput->setName(expr->name());

        auto output = tempOutput;
        return output->expr().first;
    }
};
static auto gRigister = []() {
    OnnxExtraManager::get()->insert("Upsample",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxUpSampleTransform));

    
    return true;
}();

} // namespace Express
} // namespace MNN

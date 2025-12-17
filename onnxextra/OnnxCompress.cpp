#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "config.hpp"

namespace MNN {
namespace Express {
class OnnxCompressTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int axis = 0, axisExist = 0;
        auto op = expr->get();
        for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
            auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
            auto key  = attr->key()->str();
            if (key == "axis") {
                axis = attr->i();
                axisExist = 1;
                break;
            }
        }
        VARP input = inputs[0];
        if (axisExist == 0) {
            input = _Reshape(input, {-1});
        }
        std::unique_ptr<OpT> whereOp(new OpT);
        whereOp->type = OpType_Where;
        whereOp->main.type = OpParameter_Extra;
        whereOp->main.value = new ExtraT;
        auto cond = Variable::create(Expr::create(std::move(whereOp), {inputs[1]}));
        
        auto res = _GatherV2(input, _Reshape(cond, {-1}), _Scalar<int32_t>(axis));
        res->setName(expr->name());
        return res->expr().first;
    }
};
static auto gRegister = []() {
   
    OnnxExtraManager::get()->insert("Compress", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxCompressTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
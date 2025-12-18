#include <algorithm>

#include <MNN/expr/Expr.hpp>
#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
namespace MNN {
namespace Express {
class OnnxLogSoftmaxTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        MNN_THROW_CHECK(expr->inputs().size() == 1, "Onnx LogSoftmax needs one inputs.");
        auto attrs = expr->get()->main_as_Extra()->attr();
        auto it    = std::find_if(attrs->begin(), attrs->end(),
                               [](const Attribute *attr) { return attr->key()->str() == "axis"; });
        MNN_ASSERT(it != attrs->end());
        int axis = it->i();

        VARP x           = expr->inputs()[0];
        VARP max         = _ReduceMax(x, {axis}, true);
        VARP sum         = _ReduceSum(_Exp(x - max), {axis}, true);
        VARP log         = x - max - _Log(sum);
        auto log_softmax = log->expr().first;
        log_softmax->setName(expr->name());
        return log_softmax;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("LogSoftmax",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxLogSoftmaxTransform));
    
    return true;
}();

} // namespace Express
} // namespace MNN
import os
from  pathlib import Path
import re
from prompt.utils import read_file,list_all_txt_or_py,parse_op_info_from_path
BINARY_EX_INFO = '''
CPUBinary.hpp:
//
//  CPUBinary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBinary_hpp
#define CPUBinary_hpp

#include "core/Execution.hpp"
#include "backend/cpu/CPURelu.hpp"
#include "compute/CommonOptFunction.h"
namespace MNN {
class CPUBinary : public Execution {
public:
    CPUBinary(Backend *b, MNNBinaryExecute proc, int activationType) : Execution(b) {
        mProc = proc;
        mActivationType = activationType;
    }
    virtual ~CPUBinary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static MNNBinaryExecute selectForFloat(int opType);
    static MNNBinaryExecute selectForInt(int opType);
private:
    MNNBinaryExecute mProc;
    int mNeedBroadcastIndex = -1;
    int mTotalSize;
    int mActivationType = 0;
    int mThreadNum;
    int mWorkDiv;
    std::shared_ptr<Execution> mActivationExe;
};
} // namespace MNN
#endif /* CPUBinary_hpp */

Patial content of CPUBinary.cpp:

MNNBinaryExecute CPUBinary::selectForFloat(int type) {
    auto vecFunction = selectVector<Vec4, 4, float>(type);
    if (nullptr != vecFunction) {
        return vecFunction;
    }
    switch (type) {
        case BinaryOpOperation_REALDIV:
            return execute<float, float, BinaryRealDiv<float, float, float>>;
        case BinaryOpOperation_FLOORDIV:
            return execute<float, float, BinaryFloorDiv<float, float, float>>;
        case BinaryOpOperation_FLOORMOD:
            return execute<float, float, BinaryFloorMod<float, float, float>>;
        case BinaryOpOperation_NOTEQUAL:
            return execute<float, int32_t, BinaryNotEqual<float, float, int32_t>>;
        case BinaryOpOperation_POW:
            return execute<float, float, BinaryPow<float, float, float>>;
        case BinaryOpOperation_ATAN2:
            return execute<float, float, BinaryAtan2<float, float, float>>;
        case BinaryOpOperation_MOD:
            return execute<float, float, BinaryMod<float, float, float>>;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

MNNBinaryExecute CPUBinary::selectForInt(int type) {
    auto vecFunction = selectVector<Vec4Int, 4, int32_t>(type);
    if (nullptr != vecFunction) {
        return vecFunction;
    }
    switch (type) {
        case BinaryOpOperation_MUL:
            return execute<int32_t, int32_t, BinaryMul<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_REALDIV:
            return execute<int32_t, int32_t, BinaryRealDiv<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_FLOORDIV:
            return execute<int32_t, int32_t, BinaryFloorDiv<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_FLOORMOD:
            return execute<int32_t, int32_t, BinaryFloorMod<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LOGICALOR:
            return execute<int32_t, int32_t, BinaryLogicalOr<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_NOTEQUAL:
            return execute<int32_t, int32_t, BinaryNotEqual<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_MOD:
            return execute<int32_t, int32_t, BinaryModInt<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LOGICALXOR:
            return execute<int32_t, int32_t, BinaryLogicalXor<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LEFTSHIFT:
            return execute<int32_t, int32_t, BinaryLeftShift<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_RIGHTSHIFT:
            return execute<int32_t, int32_t, BinaryRightShift<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_BITWISE_AND:
            return execute<int32_t, int32_t, BinaryBitwiseAnd<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_BITWISE_OR:
            return execute<int32_t, int32_t, BinaryBitwiseOr<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_BITWISE_XOR:
            return execute<int32_t, int32_t, BinaryBitwiseXor<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_POW:
            return execute<int32_t, int32_t, BinaryPow<int32_t, int32_t, int32_t>>;
            break;
        default:
            MNN_ERROR("Don't support binary - int compute for type %d\n", type);
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}
'''
UNARY_EX_INFO = '''
CPUUnary.hpp:
//
//  CPUUnary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUUnary_hpp
#define CPUUnary_hpp

#include "core/Execution.hpp"
#include "compute/CommonOptFunction.h"

namespace MNN {
class CPUUnary : public Execution {
public:
    CPUUnary(Backend *b, MNNUnaryExecute proc, MNNUnaryExecuteInt8 procInt8, const Op* op);
    virtual ~CPUUnary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static MNNUnaryExecute selectForFloat(int type, int precision);
    static MNNUnaryExecuteInt8 selectForInt8(int type);
protected:
    MNNUnaryExecute mProc;
    MNNUnaryExecuteInt8 mProcInt8;
    std::vector<float> mInpScale;
    std::vector<float> mOupScale;
    std::vector<ssize_t> mInpZeroPoint;
    std::vector<ssize_t> mOupZeroPoint;
    std::vector<ssize_t> mMaxMinValue;
    std::vector<int8_t> mTableBuffer;
};
} // namespace MNN
#endif /* CPUUnary_hpp */

Partial content of CPUUnary.cpp:
MNNUnaryExecute CPUUnary::selectForFloat(int type, int precision) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return _ABS;
        case UnaryOpOperation_SQUARE:
            return _Square;
        case UnaryOpOperation_NEG:
            return _Neg;
        case UnaryOpOperation_RSQRT:
            return _unaryOp<UnaryRsqrt<float>, float>;
        case UnaryOpOperation_EXP:
            return _EXP;
        case UnaryOpOperation_COS:
            return _unaryOp<UnaryCos<float>, float>;
        case UnaryOpOperation_SIN:
            return (MNNUnaryExecute)MNNSin;
        case UnaryOpOperation_SIGMOID:
            if (BackendConfig::Precision_Low == precision) {
                return (MNNUnaryExecute)MNNSigmoidLowp;
            } else {
                return (MNNUnaryExecute)MNNSigmoid;
            }
            break;
        case UnaryOpOperation_SILU:
            if (BackendConfig::Precision_Low == precision) {
                return (MNNUnaryExecute)MNNSiLuLowp;
            } else {
                return (MNNUnaryExecute)MNNSiLu;
            }
            break;
        case UnaryOpOperation_TANH:
            return (MNNUnaryExecute)MNNTanh;
        case UnaryOpOperation_TAN:
            return _unaryOp<UnaryTan<float>, float>;
        case UnaryOpOperation_ATAN:
            return _unaryOp<UnaryATan<float>, float>;
        case UnaryOpOperation_SQRT:
            return _unaryOp<UnarySqrt<float>, float>;
        case UnaryOpOperation_CEIL:
            return _unaryOp<UnaryCeil<float>, float>;
        case UnaryOpOperation_RECIPROCAL:
            return _unaryOp<UnaryRecipocal<float>, float>;
        case UnaryOpOperation_LOG1P:
            return _unaryOp<UnaryLog1p<float>, float>;
        case UnaryOpOperation_LOG:
            return _unaryOp<UnaryLog<float>, float>;
        case UnaryOpOperation_FLOOR:
            return _unaryOp<UnaryFloor<float>, float>;
        case UnaryOpOperation_BNLL:
            return _unaryOp<UnaryBNLL<float>, float>;
        case UnaryOpOperation_ACOSH:
            return _unaryOp<UnaryAcosh<float>, float>;
        case UnaryOpOperation_SINH:
            return _unaryOp<UnarySinh<float>, float>;
        case UnaryOpOperation_ASINH:
            return _unaryOp<UnaryAsinh<float>, float>;
        case UnaryOpOperation_ATANH:
            return _unaryOp<UnaryAtanh<float>, float>;
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<float>, float>;
        case UnaryOpOperation_ROUND:
            return _unaryOp<UnaryRound<float>, float>;
        case UnaryOpOperation_COSH:
            return _unaryOp<UnaryCosh<float>, float>;
        case UnaryOpOperation_ERF:
            return _unaryOp<UnaryErf<float>, float>;
        case UnaryOpOperation_ERFC:
            return _unaryOp<UnaryErfc<float>, float>;
        case UnaryOpOperation_ERFINV:
            return _unaryOp<UnaryErfinv<float>, float>;
        case UnaryOpOperation_EXPM1:
            return _EXPM1;
        case UnaryOpOperation_ASIN:
            return _unaryOp<UnaryAsin<float>, float>;
        case UnaryOpOperation_ACOS:
            return _unaryOp<UnaryAcos<float>, float>;
        case UnaryOpOperation_HARDSWISH:
            return (MNNUnaryExecute)MNNHardSwishCommon;
        case UnaryOpOperation_GELU:
            return (MNNUnaryExecute)MNNGeluCommon;
        case UnaryOpOperation_GELU_STANDARD:
            return (MNNUnaryExecute)MNNGeluStandardCommon;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

'''
REDUCTION_CPP='''
//
//  CPUReduction.cpp
//  MNN
//
//  Created by MNN on 2018/07/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUReduction.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include <cmath>
#include <algorithm>
#include "core/OpCommonUtils.hpp"
#define UNIT 4
#define UNIT_DUP(value) \
    { (value), (value), (value), (value) }

namespace MNN {
// outside, axis, inside

class Reduction : public Execution {
public:
    Reduction(Backend* backend, const Op* op) : Execution(backend) {
        // Do nothing
        mAxis = op->main_as_ReductionParam()->dim()->data()[0];
    }
    virtual ~Reduction() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto input  = inputs[0];
        auto output = outputs[0];
        auto typeCode = input->getType().code;
        auto src = inputs[0];
        int outside = 1;
        for(int i=0; i<mAxis; ++i) {
            outside *= input->length(i);
        }
        int inside = 1;
        for(int i=mAxis+1; i<input->dimensions(); ++i) {
            inside *= input->length(i);
        }
        auto axis = input->length(mAxis);
        auto dst = output;
        //MNN_ASSERT(output->elementSize() == inside * outside);
        if (halide_type_float == typeCode) {
            this->onReduce(src->host<float>(), dst->host<float>(), inside, outside, axis);
        } else if (halide_type_int == typeCode) {
            this->onReduce(src->host<int32_t>(), dst->host<int32_t>(), inside, outside, axis);
        }
        return NO_ERROR;
    }
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axis) const     = 0;
    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outsize, int axis) const = 0;
private:
    int mAxis = -1;
};

class MeanReduce : public Reduction {
public:
    MeanReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MeanReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside % 4 == 0) {
                    ::memcpy(dstOutSide, srcOutSide, inside * sizeof(float));
                    for (int a = 1; a < axisSize; ++a) {
                        auto srcAxis = srcOutSide + a * inside;
                        MNNMatrixAddCommon(dstOutSide, dstOutSide, srcAxis, inside, 0, 0, 0, 1);
                    }
                    float divide = 1.0f / (float)axisSize;
                    for (int i=0; i<inside; ++i) {
                        dstOutSide[i] = dstOutSide[i] * divide;
                    }
                } else {
                    for (int ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutSide + ii;
                        auto dstInside = dstOutSide + ii;
                        float summer   = 0.0f;
                        for (int a = 0; a < axisSize; ++a) {
                            summer += srcInside[a * inside];
                        }
                        *dstInside = summer / (float)axisSize;
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += srcInside[a * inside];
                }
                *dstInside = summer / axisSize;
            }
        }
    }
};

class SumReduce : public Reduction {
public:
    SumReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~SumReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        auto core = static_cast<CPUBackend*>(backend())->functions();
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            for (int oi = tId; oi < outside; oi+=numberThread) {
                auto srcOutSide = src + oi * axisSize * inside;
                auto dstOutSide = dst + oi * inside;
                if (inside % 4 == 0) {
                    ::memcpy(dstOutSide, srcOutSide, inside * sizeof(float));
                    for (int a = 1; a < axisSize; ++a) {
                        auto srcAxis = srcOutSide + a * inside;
                        MNNMatrixAddCommon(dstOutSide, dstOutSide, srcAxis, inside, 0, 0, 0, 1);
                    }
                } else {
                    for (int ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutSide + ii;
                        auto dstInside = dstOutSide + ii;
                        float summer   = 0.0f;
                        if (inside == 1) {
                            core->MNNAccumulateSequenceNumber(&summer, srcInside, axisSize);
                        } else {
                            for (int a = 0; a < axisSize; ++a) {
                                summer += srcInside[a * inside];
                            }
                        }
                        *dstInside = summer;
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t summer = 0;
                for (int a = 0; a < axisSize; ++a) {
                    summer += srcInside[a * inside];
                }
                *dstInside = summer;
            }
        }
    }
};

class MinReduce : public Reduction {
public:
    MinReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MinReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float Min      = srcInside[0];
                if (1 == inside) {
                    int32_t inputCountUnit = axisSize / (UNIT * 2);
                    int32_t remain         = axisSize - (inputCountUnit * UNIT * 2);
                    float minArray[UNIT]   = UNIT_DUP(Min);
                    MNNMinFloat((float*)srcInside, minArray, inputCountUnit);

                    for (int i = 0; i < UNIT; i++) {
                        Min = std::min(Min, minArray[i]);
                    }
                    if (remain > 0) {
                        int currentIndex = inputCountUnit * UNIT * 2;
                        for (int i = 0; i < remain; i++) {
                            float currentInputData = srcInside[currentIndex + i];
                            Min                    = std::min(Min, currentInputData);
                        }
                    }
                } else {
                    for (int a = 0; a < axisSize; ++a) {
                        Min = std::min(Min, srcInside[a * inside]);
                    }
                }
                *dstInside = Min;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t Min    = srcInside[0];
                for (int a = 0; a < axisSize; ++a) {
                    Min = std::min(Min, srcInside[a * inside]);
                }
                *dstInside = Min;
            }
        }
    }
};

class MaxReduce : public Reduction {
public:
    MaxReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~MaxReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float Max      = srcInside[0];
                if (1 == inside && axisSize > UNIT * 2) {
                    int32_t inputCountUnit = axisSize / (UNIT * 2);
                    int32_t remain         = axisSize - (inputCountUnit * UNIT * 2);
                    float maxArray[UNIT]   = UNIT_DUP(Max);

                    MNNMaxFloat((float*)srcInside, maxArray, inputCountUnit);

                    for (int i = 0; i < UNIT; i++) {
                        Max = std::max(Max, maxArray[i]);
                    }
                    if (remain > 0) {
                        int currentIndex = inputCountUnit * UNIT * 2;
                        for (int i = 0; i < remain; i++) {
                            float currentInputData = srcInside[currentIndex + i];
                            Max                    = std::max(Max, currentInputData);
                        }
                    }
                } else {
                    for (int a = 0; a < axisSize; ++a) {
                        Max = std::max(Max, srcInside[a * inside]);
                    }
                }
                *dstInside = Max;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                int32_t Max    = srcInside[0];
                for (int a = 0; a < axisSize; ++a) {
                    Max = std::max(Max, srcInside[a * inside]);
                }
                *dstInside = Max;
            }
        }
    }
};

class ProdReduce : public Reduction {
public:
    ProdReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ProdReduce() = default;

protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside = srcOutSide + ii;
                auto dstInside = dstOutSide + ii;
                float product  = 1.0f;
                for (int a = 0; a < axisSize; ++a) {
                    product *= srcInside[a * inside];
                }
                *dstInside = product;
            }
        }
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t product = 1;
                for (int a = 0; a < axisSize; ++a) {
                    product *= srcInside[a * inside];
                }
                *dstInside = product;
            }
        }
    }
};

class AnyReduce : public Reduction {
public:
    AnyReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ AnyReduce() = default;
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        MNN_ASSERT(false);
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t result = 0;
                for (int a = 0; a < axisSize; ++a) {
                    if (srcInside[a * inside] > 0) {
                        result = 1;
                        break;
                    }
                }
                *dstInside = result;
            }
        }
    }
};

class AllReduce : public Reduction {
public:
    AllReduce(Backend* backend, const Op* op) : Reduction(backend, op) {
        // nothing to do
    }
    virtual ~ AllReduce() = default;
protected:
    virtual void onReduce(const float* src, float* dst, int inside, int outside, int axisSize) const override {
        MNN_ASSERT(false);
    }

    virtual void onReduce(const int32_t* src, int32_t* dst, int inside, int outside, int axisSize) const override {
        for (int oi = 0; oi < outside; ++oi) {
            auto srcOutSide = src + oi * axisSize * inside;
            auto dstOutSide = dst + oi * inside;
            for (int ii = 0; ii < inside; ++ii) {
                auto srcInside  = srcOutSide + ii;
                auto dstInside  = dstOutSide + ii;
                int32_t result = 1;
                for (int a = 0; a < axisSize; ++a) {
                    if (srcInside[a * inside] == 0) {
                        result = 0;
                        break;
                    }
                }
                *dstInside = result;
            }
        }
    }
};

Execution* CPUReductionCreator::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) const {
    return create(inputs, outputs, op, backend);
}

Execution* CPUReductionCreator::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const MNN::Op* op, Backend* backend) {
    auto type = inputs[0]->getType();
    if (type.bits != 32) {
        return nullptr;
    }
    if (type.code != halide_type_float && type.code != halide_type_int) {
        return nullptr;
    }
    switch (op->main_as_ReductionParam()->operation()) {
        case ReductionType_MEAN:
            return new MeanReduce(backend, op);
        case ReductionType_SUM:
            return new SumReduce(backend, op);
        case ReductionType_MINIMUM:
            return new MinReduce(backend, op);
        case ReductionType_MAXIMUM:
            return new MaxReduce(backend, op);
        case ReductionType_PROD:
            return new ProdReduce(backend, op);
        case ReductionType_ANY:
            return new AnyReduce(backend, op);
        case ReductionType_ALL:
            return new AllReduce(backend, op);
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

REGISTER_CPU_OP_CREATOR(CPUReductionCreator, OpType_Reduction);

} // namespace MNN
'''
REDUCTION_EX_INFO = '''
CPUReduction.hpp:
//
//  CPUReduction.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUReduction_hpp
#define CPUReduction_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class CPUReductionCreator : public CPUBackend::Creator {
public:
    static Execution* create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                             const MNN::Op* op, Backend* backend);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override;
};
} // namespace MNN
#endif /* CPUReduction_hpp */
'''
CONV_EX_INFO = {
    "Dense":'''
//
//  DenseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DenseConvolutionTiledExecutor_hpp
#define DenseConvolutionTiledExecutor_hpp


#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
#include "ConvolutionTiledExecutor.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
typedef void(*lowMemoryMatmulUnit)(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
typedef void(*lowMemoryMatmulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
class DenseConvolutionTiledImpl : public ConvolutionTiledImpl {
public:
    DenseConvolutionTiledImpl(const Convolution2DCommon *common, Backend *b, CPUConvolution::Resource* resource = nullptr) : ConvolutionTiledImpl(common, b) {
        mResource = resource;
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    virtual ~DenseConvolutionTiledImpl() = default;
    void getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) override;
    static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b);
protected:
};
class DenseConvolutionTiledExecutor : public ConvolutionTiledExecutor {
public:
    DenseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                  size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common>);

    DenseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon *common, Backend* b);
    virtual ~DenseConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function);
    static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
        return DenseConvolutionTiledImpl::bestTileConvolutionConfig(common, inputTensor, outputTensor, threadNumber, b);
    }
    static bool initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes);
    static void selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const CoreFunctions* core);
    struct DequantizeCache {
        std::shared_ptr<MNN::Tensor> weight;
        std::shared_ptr<MNN::Tensor> weightInt8;
    };
protected:
    DequantizeCache mWeightCache;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
};

class ConvolutionTiledExecutorMultiInput : public Execution {
public:
    ConvolutionTiledExecutorMultiInput(const Convolution2DCommon *common, Backend *b) : Execution(b) {
        mProxy.reset(new DenseConvolutionTiledImpl(common, b));
    }
    virtual ~ConvolutionTiledExecutorMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempWeightCache;
    std::shared_ptr<Tensor> mTempBias;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
    std::vector<Tensor *> mInputs;
};
} // namespace MNN

#endif /* DenseConvolutionTiledExecutor_hpp */

''',
    "Strassen":'''
//
//  Convolution1x1Strassen.hpp
//  MNN
//
//  Created by MNN on 2019/02/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Convolution1x1Strassen_hpp
#define Convolution1x1Strassen_hpp

#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/StrassenMatmulComputor.hpp"
namespace MNN {
#ifndef MNN_REDUCE_SIZE
class Convolution1x1Strassen : public CPUConvolution {
public:
    Convolution1x1Strassen(const Convolution2DCommon *common, Backend *b, const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize);
    Convolution1x1Strassen(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b);
    virtual ~Convolution1x1Strassen();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    std::shared_ptr<CPUConvolution::Resource> mResource;

    struct Unit {
        bool mValid = true;
        int offset[4];//Input, Weight, Output, Bias
        std::shared_ptr<StrassenMatrixComputor> mStracssenComputor;
    };

    std::vector<Unit> mUnits;
    int mWeightBytes = 4;
};
#endif
} // namespace MNN

#endif /* Convolution1x1Strassen_hpp */

''',
    "Winograd":'''
//
//  ConvolutionPackWinograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionPackWinograd_hpp
#define ConvolutionPackWinograd_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/ConvolutionWinogradImpl.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2

namespace MNN {
class ConvolutionPackWinograd : public ConvolutionWinogradImpl {
public:
    ConvolutionPackWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output, Backend *b,
                        const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize,
                        WinogradConfig config);
    virtual ~ConvolutionPackWinograd();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static WinogradConfig bestWinogradUnit(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                int threadnumber, Backend* b, const PerfConfig& denseConfig);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ConvolutionPackWinograd(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *convOp, Backend* b)
    : ConvolutionWinogradImpl(convOp, b) {
        mResource = resource;
    }
    std::pair<int, std::function<void(int tId, const uint8_t*, uint8_t*)>> mMainFunction;
    std::pair<int, std::function<void(int, uint8_t*)>> mPostFunction;

};
} // namespace MNN
#endif /* ConvolutionPackWinograd_hpp */
''',
    "Group":'''
//
//  ConvolutionGroup.hpp
//  MNN
//
//  Created by MNN on 2018/08/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionGroupInt8_hpp
#define ConvolutionGroupInt8_hpp

#include "backend/cpu/compute/ConvolutionIntFactory.hpp"

namespace MNN {
class ConvolutionGroup : public Execution {
public:
    ConvolutionGroup(Backend *b, const std::vector<std::shared_ptr<Execution>> &subConvolution);
    virtual ~ConvolutionGroup() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::unique_ptr<Tensor> mInputRaw;
    std::unique_ptr<Tensor> mOutputRaw;

    std::unique_ptr<Tensor> mInputUnit;
    std::unique_ptr<Tensor> mOutputUnit;

    std::vector<Tensor *> mInputUnitWrap;
    std::vector<Tensor *> mOutputUnitWrap;
    std::vector<std::shared_ptr<Execution>> mSubConvolution;
};
} // namespace MNN

#endif /* ConvolutionGroupInt8_hpp */

'''
}

FEW_SHOT_PMPT= '''
You are an expert in model deployment, proficient in PyTorch and C++ programming, and familiar with the coding style of the MNN framework. You will be given a PyTorch model which will be exported as an ONNX graph and then converted into an MNN computation graph. Your task is to write C++ code that implements and accelerates the operators from this model for MNN's CPU backend. Note that:

- Understand the example thoroughly and think carefully before you write the code.
- Provide only the code file as your final answer, without any explanations or comments.
- Your code must adhere to the supported API surfaces, invoking only official functions and members when using MNN C++ interfaces( e.g. Math, Tensor, VARP, Matrix) and flatbuffers library.
- Each file name must appear as the very first line inside its corresponding code block. If provide cpu backend implement, provide the hpp code and cpp code seperately. 
- Implement the operator’s computation logic in a self-contained manner. Minimize coupling to MNN internals and 3rd party libraries; call APIs only when strictly required.
- Implement methods include: the CPU backend implement which handles numerical computation for operators by memory management and instruction-level optimization, the geometry computation which manages data layout and memory mapping and is used for operators that change tensor shapes or memory arrangements, the combinator implementation which builds new operator functions by composing existing MNN operators.
- When write CPU backend operator, implement onResize and onExecute. In onResize, allocate the cache buffer using backend()->onAcquireBuffer(&mCache, Backend::DYNAMIC) and release it with backend()->onReleaseBuffer(&mCache, Backend::DYNAMIC), allowing the freed memory to be reused. In onExecute, perform necessary input validation to catch issues early. Return NO_ERROR upon successful execution.
- When write a Geometry backend operator, implement onCompute to construct the tensor regions and command sequence that describe how outputs are assembled from inputs, allocating any intermediate tensors as virtual slices so the runtime can later schedule the actual computation efficiently.
- When write a Combiner operator, implement the onExecute method which parses the ONNX node's inputs and attributes to construct an equivalent computational subgraph using MNN's Express API and returns the converted expression to complete the operator translation.

Here are some examples:
===== Example 1 Start =====
Given PyTorch model Det: \n
``` \n
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a summation using Det.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Det summation to the input tensor.
        Sums over the last two dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, M).

        Returns:
            torch.Tensor: Output tensor of shape (B,), where each element
                          is the sum over the last two dimensions of x.
        """
        # Det expression: "b n m -> b" means sum over n & m for each batch
        return torch.det(x.transpose(-1, -2).matmul(x)).sum(dim=-1)
    
M = 256
N = 256
batch_size = 16

def get_inputs():
    x = torch.rand(batch_size, N, M)
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed
```\n
By using CPU backend implemetation, you should respond with the final answer with two files seperately:
CPUDet.hpp:
``` \n
//
//  CPUDet.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDet_hpp
#define CPUDet_hpp

#include <MNN/Tensor.hpp>
#include "core/Execution.hpp"

namespace MNN {
class CPUDet : public Execution {
public:
    CPUDet(Backend *bn) : Execution(bn) { }
    virtual ~CPUDet() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempMat, mTempRowPtrs;
};

} // namespace MNN
#endif /* CPUDet_hpp */
``` \n
CPUDet.cpp:
``` \n
//
//  CPUDet.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include <limits>
#include "CPUDet.hpp"
#include "CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {
ErrorCode CPUDet::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    auto M = inputs[0]->length(1);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mTempMat.reset(Tensor::createDevice<float>({numberThread, M, ROUND_UP(M, core->pack)}));
    mTempRowPtrs.reset(Tensor::createDevice<float*>({numberThread, M}));
    auto success = backend()->onAcquireBuffer(mTempMat.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mTempRowPtrs.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempMat.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempRowPtrs.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUDet::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto input  = inputs[0], output = outputs[0];
    auto batch = input->length(0), M = input->length(1), step = ROUND_UP(M, core->pack);
    auto computeDet = [&](int b, int tId) -> float {
#define F_IS_ZERO(v) (fabs(v) < 1e-6)
#define ADDR(row) (mTempRowPtrs->host<float*>()[tId * M + row])
#define VAL(row, col) (*(ADDR(row) + col))
        auto elimRow = [&](int row1, int row2) {
            auto ratio = -VAL(row2, row1) / VAL(row1, row1);
            float params[] = {1.f, ratio, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()};
            int sta = row1, end = M;
            int extra = (core->pack - (end - sta) % core->pack) % core->pack;
            if (step - M >= extra) {
                end = M + extra;
            } else {
                sta -= extra - (step - M);
                end = step;
            }
            auto p1 = ADDR(row1) + sta, p2 = ADDR(row2) + sta;
            core->MNNAxByClampBroadcastUnit(p2, p2, p1, 1, core->pack, core->pack, (end - sta) / core->pack, params);
        };
        float result = 1;
        for (int i = 0; i < M; ++i) {
            auto tempPtr = mTempMat->host<float>() + (tId * M + i) * step;
            ::memcpy(tempPtr, input->host<float>() + (b * M + i) * M, M * sizeof(float));
            mTempRowPtrs->host<float*>()[tId * M + i] = tempPtr;
        }
        for (int i = 0; i < M; ++i) {
            if (F_IS_ZERO(VAL(i, i))) {
                bool swapd = false;
                for (int j = i + 1; j < M; ++j) {
                    if (!F_IS_ZERO(VAL(j, i))) {
                        std::swap(ADDR(i), ADDR(j));
                        swapd = true;
                        break;
                    }
                }
                if (!swapd) {
                    return 0;
                }
            }
            result *= VAL(i, i);
            for (int j = i + 1; j < M; ++j) {
                elimRow(i, j);
            }
        }
        return result;
    };
    
    int numberThread = ((CPUBackend*)backend())->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int b = tId; b < batch; b += numberThread) {
            output->host<float>()[b] = computeDet(b, tId);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
class CPUDetCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUDet(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDetCreator, OpType_Det);
} // namespace MNN
``` \n
===== Example 1 End =====

===== Example 2 Start =====
Given PyTorch model Gather: \n
``` \n
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a summation using Combiner gather.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Combiner gather summation to the input tensor.
        Sums over the last two dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, M).

        Returns:
            torch.Tensor: Output tensor of shape (B,), where each element
                          is the sum over the last two dimensions of x.
        """
        # Combiner gather expression: "b n m -> b" means sum over n & m for each batch
        return torch.sum(x, dim=(-2, -1))
    
batch_size = 64
N = 128
M = 256

def get_inputs():
    x = torch.rand(batch_size, N, M)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
``` \n

By using combiner implemetation, you should respond with the final answer:\n
OnnxGather.cpp:
```
//
//  OnnxGather.cpp
//  MNNConverter
//
//  Created by MNN on 2020/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"
#include "config.hpp"

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
class OnnxGatherTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int axis    = 0;
        auto op     = expr->get();
        if (nullptr != op->main_as_Extra()->attr()) {
            for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                    break;
                }
            }
        }
        auto axisVar = _Scalar<int>(axis);
        auto config = Global<modelConfig>::Get();
        if (config->optimizeLevel < 2) {
            // Add negative protect, may decrease performance
            auto rankVar = _Rank(inputs[0]);
            axisVar = _Mod(axisVar + rankVar, rankVar);
            auto shapeVar = _Shape(inputs[0], true);
            auto axisLengthVar = _Squeeze(_StridedSlice(shapeVar, _Unsqueeze(axisVar, {0}), _Unsqueeze(axisVar + _Scalar<int>(1), {0}),  _Unsqueeze(_Scalar<int32_t>(1), {0}), 0, 0, 0, 0, 0));
            inputs[1] = _Mod(inputs[1] + axisLengthVar, axisLengthVar);
        }
        auto output = _GatherV2(inputs[0], inputs[1], axisVar);
        output->setName(expr->name());
        return output->expr().first;
    }
};

class OnnxGatherNDTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        auto indice = inputs[1];
        auto param = inputs[0];
        std::unique_ptr<OpT> op(new OpT);
        op->type = OpType_GatherND;
        // Default is 0, Ref https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
        int batch_dims    = 0;
        auto oriop     = expr->get();
        if (nullptr != oriop->main_as_Extra()->attr()) {
            for (int i = 0; i < oriop->main_as_Extra()->attr()->size(); ++i) {
                auto attr = oriop->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "batch_dims") {
                    batch_dims = attr->i();
                    break;
                }
            }
        }
        if (batch_dims != 0) {
            op->main.value = new AxisT;
            op->main.type = OpParameter_Axis;
            op->main.AsAxis()->axis = batch_dims;
            // Add Extra offset for indice
            auto indiceShape = _Shape(indice, true);
            auto startV = _Unsqueeze(_Scalar<int>(0), {0});
            auto batchV = _Unsqueeze(_Scalar<int>(batch_dims), {0});
            batchV.fix(VARP::CONSTANT);
            startV.fix(VARP::CONSTANT);
            auto rankIndice = _Unsqueeze(_Rank(indice), {0});
            auto totalSize = _Slice(indiceShape, startV, batchV);
            auto start = _Scalar<int>(0);
            auto delta = _Scalar<int>(1);
            // Make offset from range: [B, 1]
            auto offset = _Range(start, _ReduceProd(totalSize), delta);
            VARP dims = rankIndice - batchV;
            auto oneSize = _Fill(dims, _Scalar<int>(1));
            offset = _ReshapeF(offset, _Concat({totalSize, oneSize}, 0), MNN::MNN_DATA_FORMAT_NCHW);

            // Compute Stride
            auto lastDim = _Slice(indiceShape, rankIndice - delta, _Unsqueeze(delta, {0}));
            auto paramShape = _Shape(inputs[0], true);
            paramShape->setName(inputs[0]->name() + "_shape");
            auto rankInput = _Unsqueeze(_Rank(inputs[0]), {0});
            auto inputDimOffset = rankInput - batchV;
            auto inputSDim = inputDimOffset - lastDim;
            auto lastInputShapes = _Slice(paramShape, batchV, rankInput - batchV - inputSDim);
            auto strideParameter = _ReduceProd(lastInputShapes);
            strideParameter->setName(param->name() + "_stride");
            offset = offset * strideParameter;
            indice = indice + offset;
        }
        auto outputExpr = Expr::create(op.get(), {param, indice}, 1);
        outputExpr->setName(expr->name());
        return outputExpr;
    }
};

class OnnxGatherElementTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        int axis    = 0;
        auto op     = expr->get();
        if (nullptr != op->main_as_Extra()->attr()) {
            for (int i = 0; i < op->main_as_Extra()->attr()->size(); ++i) {
                auto attr = op->main_as_Extra()->attr()->GetAs<Attribute>(i);
                auto key  = attr->key()->str();
                if (key == "axis") {
                    axis = attr->i();
                    break;
                }
            }
        }
        if (inputs.size() < 2) {
            MNN_ERROR("GatherElements should has two inputs\n");
            return nullptr;
        }
        // Reshape the input as outside, axis, inside
        auto index = inputs[1];
        auto input = inputs[0];
        auto dst = Express::_GatherElements(input, index, _Scalar(axis));
        dst->setName(expr->name());
        return dst->expr().first;
    }
};

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
    OnnxExtraManager::get()->insert("Gather", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherTransform));
    OnnxExtraManager::get()->insert("GatherND", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherNDTransform));
    OnnxExtraManager::get()->insert("GatherElements", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxGatherElementTransform));
    OnnxExtraManager::get()->insert("Compress", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxCompressTransform));
    return true;
}();

} // namespace Express
} // namespace MNN
``` \n
===== Example 2 End =====

===== Example 3 Start =====

Given PyTorch model Det: \n
``` \n
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a summation using Det.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Det summation to the input tensor.
        Sums over the last two dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, M).

        Returns:
            torch.Tensor: Output tensor of shape (B,), where each element
                          is the sum over the last two dimensions of x.
        """
        # Det expression: "b n m -> b" means sum over n & m for each batch
        return torch.det(x.transpose(-1, -2).matmul(x)).sum(dim=-1)
    
M = 256
N = 256
batch_size = 16

def get_inputs():
    x = torch.rand(batch_size, N, M)
    return [x]
def get_init_inputs():
    return []  # No special initialization inputs needed
``` \n
By using geometry implemetation,  you should respond with the final answer:\n
GeometryDet.cpp:
```\n
//
//  GeometryDet.cpp
//  MNN
//
//  Created by MNN on 2020/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
namespace MNN {
class GeometryDet : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto input = inputs[0], output = outputs[0];
        auto batch = output->elementSize(), M = input->length(input->dimensions() - 1);
        
        auto midInput = std::shared_ptr<Tensor>(Tensor::createDevice({batch, M, M}, input->getType(), input->getDimensionType()));
        auto midInDes = TensorUtils::getDescribe(midInput.get());
        midInDes->regions = {TensorUtils::makeFullSlice(input)};
        midInDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        
        auto midOutput = std::shared_ptr<Tensor>(Tensor::createDevice({batch}, output->getType(), output->getDimensionType()));
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        outDes->regions = {TensorUtils::makeFullSlice(midOutput.get())};
        outDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        
        std::shared_ptr<Command> cmd(new Command);
        cmd->op = op;
        cmd->inputs.assign({midInput.get()});
        cmd->outputs.assign({midOutput.get()});
        res.command.emplace_back(std::move(cmd));
        
        res.extras.emplace_back(std::move(midInput));
        res.extras.emplace_back(std::move(midOutput));
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryDet);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Det});
}

REGISTER_GEOMETRY(GeometryDet, _create)

} // namespace MNN
``` \n
===== Example 3 End =====

Now you are given the following PyTorch model:
``` \n
__TARGET_MODEL__
```\n
Here is the extra information:
```\n
__EXTRA_INFO__
```\n
'''


REPO_TOP_PATH = ""
###############################################################################
# Single kernel problem prompt generator #########
##############################################################################
SYSTEMROLE = """You are a mobile model deployment expert, specializing in operator development and optimization for the MNN framework. Your task is to optimize operators under the CPU backend of MNN, either to extend the operator support or achieve speedups.\n"""
BACKGROUNDINFO = "In a standard deployment pipeline, PyTorch models are first converted to ONNX and then to MNN. Your goal is to implement MNN support for the corresponding ONNX operators. You may choose to: Implement a custom CPU backend operator by providing CPUMyCustomOp.hpp and CPUMyCustomOp.cpp. Define a compositor that reuses and combines existing MNN operators by providing OnnxMyCustomOp.cpp. Use geometry computation to express the operator’s functionality through tensor shape and layout transformations by providing GeometryMyCustomOp.cpp.You are encouraged to select the most effective and efficient implementation strategy based on the operator’s behavior and performance characteristics."

def oneshot_prompt_generator_for_MNN(
    targ_model: str, example_model: str, example_MNN_OP: str,reference_head:str,
        op_name:str,OP_type : str= "CPU"
) -> str:
    
    
    USER_prompt_template = f"""You are an expert in model deployment, proficient in PyTorch and C++ programming, and familiar with the coding style of the MNN framework. You will be given a PyTorch model which will be exported as an ONNX graph and then converted into an MNN computation graph. Your task is to write C++ code that implements and accelerates the operators from this model for MNN's CPU backend. Note that:

- Think carefully before providing your final answer.
- Your code must adhere to the supported API surfaces, invoking only official functions and members when using MNN C++ interfaces( e.g. Math, Tensor, VARP, Matrix) and flatbuffers library.
- Implement the operator’s core computation logic in a self-contained manner. Minimize coupling to MNN internals and 3rd party libraries; call APIs only when strictly required.
- Implement methods include: the CPU backend implement which handles numerical computation for operators by memory management and instruction-level optimization, the geometry computation which manages data layout and memory mapping and is used for operators that change tensor shapes or memory arrangements, the combinator implementation which builds new operator functions by composing existing MNN operators.
- File name rule: given an operator named OP, then a CPU backend implementation must provide CPUOp.hpp and CPUOp.cpp files separately, if implemented as a combinator, provide OnnxOp.cpp file, if implemented via geometry computation, provide GeometryOp.cpp file.
- Code content rule: Each file name must appear as the very first line inside its corresponding code block. If provide cpu backend implement, organize the code content according to the given example. 
- Implement onResize and onExecute. In onResize, allocate the cache buffer using backend()->onAcquireBuffer(&mCache, Backend::DYNAMIC) and release it with backend()->onReleaseBuffer(&mCache, Backend::DYNAMIC), allowing the freed memory to be reused. In onExecute, perform necessary input validation to catch issues early. Return NO_ERROR upon successful execution.

"""
    final_request = ""
    prompt = USER_prompt_template

    # achievement = {
    #     "CPU": "Implement a custom CPU backend operator by providing CPUMyCustomOp.hpp_like and CPUMyCustomOp.cpp_like files",
    #     "Combiner": "Define a compositor that reuses and combines existing MNN operators by providing OnnxMyCustomOp.cpp_like file",
    #     "Geometry": "Use geometry computation to express the operator by providing GeometryMyCustomOp.cpp_like file",
    # }
    achievement = {
        "CPU": "implement a custom CPU backend operator by providing CPUMyCustomOp.hpp_like and CPUMyCustomOp.cpp_like files",
        "Combiner": "define a combinator that reuses and combines existing MNN operators by providing OnnxMyCustomOp.cpp_like file",
        "Geometry": "use geometry computation to express the operator by providing GeometryMyCustomOp.cpp_like file",
    }
    
    # problem_instruction = f"""Based on the following PyTorch model example, choose the most efficient approach to support the operator in the target model and implement computation acceleration. Only provide the file contents corresponding to your chosen implementation method, giving code but no any annotations."""
#
    # problem_instruction = f"""Based on the following PyTorch model example, {achievement[OP_type]} to achieve speedups. Only provide the file contents corresponding to your chosen implementation method, giving code but no any annotations."""


    if example_model != "" and example_MNN_OP != "":
        prompt += f"""Here are some basic example:
``` \n

===== Example  Start =====

Given PyTorch model the implementation:
``` \n
{example_model}
``` \n
You should respond with: 

Then the final answer is:
```
{example_MNN_OP}
``` \n
===== Example  End =====
    """

    prompt += f"""
Now you are given the following PyTorch model, with name {op_name}:
``` \n
{targ_model}
```
Here is the reference but not limited  head file:

```
{reference_head}
```
    """
    prompt += f" {achievement[OP_type]}, and only respond with your final answer."
    return prompt

def prompt_generator_for_single_OP_from_prompt_template(target_model: str,
                                                        example_folder:str,
                                                        op_name:str,
                                                        referencehead:str="",
                                                        backend = "CPU",
                                                        OP_type= "CPU",
                                                        ) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    for folder in os.listdir(example_folder):
        if "cpu" in folder:
            cpu_folder = os.path.join(example_folder, folder)
            for file in os.listdir(cpu_folder):
                if file.endswith(".py"):
                    example_cpu_model = read_file(os.path.join(example_folder,cpu_folder, file))
                if file.endswith(".hpp"):
                    hpp = read_file(os.path.join(example_folder,cpu_folder, file))
                if file.endswith(".cpp"):
                    cpp = read_file(os.path.join(example_folder,cpu_folder, file))
            example_cpu_MNN_OP = hpp + cpp
        if "combiner" in folder:
            combiner_folder = os.path.join(example_folder, folder)
            for file in os.listdir(combiner_folder):
                if file.endswith(".py"):
                    example_combiner_model = read_file(os.path.join(example_folder,combiner_folder, file))
                if file.endswith(".cpp"):
                    example_combiner_MNN_OP = read_file(os.path.join(example_folder,combiner_folder, file))
        else:
            example_combiner_MNN_OP = ""
        if "geometry" in folder:
            geometry_folder = os.path.join(example_folder, folder)
            for file in os.listdir(geometry_folder):
                if file.endswith(".py"):
                    example_geo_model = read_file(os.path.join(example_folder,geometry_folder, file))
                if file.endswith(".cpp"):
                    example_geo_MNN_OP = read_file(os.path.join(example_folder,geometry_folder, file))
        else:
            example_geo_MNN_OP = ""
    match OP_type:
        case "CPU":
            example_model = example_cpu_model
            example_MNN_OP = example_cpu_MNN_OP
        case "Combiner":
            example_model = example_combiner_model
            example_MNN_OP = example_combiner_MNN_OP
        case "Geometry":
            example_model = example_geo_model
            example_MNN_OP = example_geo_MNN_OP
    return oneshot_prompt_generator_for_MNN(target_model,
                                             example_model, example_MNN_OP,
                                            op_name = op_name,
                                            reference_head=referencehead,
                                            backend=backend,
                                            OP_type=OP_type)

def prompt_generator_for_single_OP_from_fewshot_prompt_template(
    target_model: str,example_folder:str,op_name:str,referencehead:str="",
    backend: str = "CPU",OP_type: str = "CPU compute"
) -> str:
    """
    Using few-shot prompt template with multiple examples for different implementation methods
    """
    # extract example models and MNN OP code from example folder
    for folder in os.listdir(example_folder):
        if "cpu" in folder:
            cpu_folder = os.path.join(example_folder, folder)
            for file in os.listdir(cpu_folder):
                if file.endswith(".py"):
                    example_cpu_model = read_file(os.path.join(example_folder,cpu_folder, file))
                if file.endswith(".hpp"):
                    hpp = read_file(os.path.join(example_folder,cpu_folder, file))
                if file.endswith(".cpp"):
                    cpp = read_file(os.path.join(example_folder,cpu_folder, file))
            example_cpu_MNN_OP = hpp + cpp
        if "combiner" in folder:
            combiner_folder = os.path.join(example_folder, folder)
            for file in os.listdir(combiner_folder):
                if file.endswith(".py"):
                    example_combiner_model = read_file(os.path.join(example_folder,combiner_folder, file))
                if file.endswith(".cpp"):
                    example_combiner_MNN_OP = read_file(os.path.join(example_folder,combiner_folder, file))

        if "geometry" in folder:
            geometry_folder = os.path.join(example_folder, folder)
            for file in os.listdir(geometry_folder):
                if file.endswith(".py"):
                    example_geo_model = read_file(os.path.join(example_folder,geometry_folder, file))
                if file.endswith(".cpp"):
                    example_geo_MNN_OP = read_file(os.path.join(example_folder,geometry_folder, file))

    return fewshot_prompt_generator_for_MNN(
        target_model,
        example_cpu_model,
        example_cpu_MNN_OP,
        example_combiner_model,
        example_combiner_MNN_OP,
        example_geo_model,
        example_geo_MNN_OP,
        op_name = op_name,
        reference_head=referencehead,
        backend=backend,
        OP_type=OP_type
    )


def fewshot_prompt_generator_for_MNN(
    targ_model: str, 
    example_cpu_model: str, example_cpu_MNN_OP: str,
      example_combiner_model: str, example_combiner_MNN_OP: str,
        example_geo_model: str, example_geo_MNN_OP: str,
        op_name:str,reference_head:str,
        backend: str = "CPU", OP_type: str = "CPU"
) -> str:

    USER_prompt_template = f"""You are an expert in model deployment, proficient in PyTorch and C++ programming, and familiar with the coding style of the MNN framework. You will be given a PyTorch model which will be exported as an ONNX graph and then converted into an MNN computation graph. Your task is to write C++ code that implements and accelerates the operators from this model for MNN's CPU backend. Note that:

- Think carefully before providing your final answer, then only response with final answer.
- Your code must adhere to the supported API surfaces, invoking only official functions and members when using MNN C++ interfaces( e.g. Math, Tensor, VARP, Matrix) and flatbuffers library.
- Implement the operator’s core computation logic in a self-contained manner. Minimize coupling to MNN internals and 3rd party libraries; call APIs only when strictly required.
- Implement methods include: the CPU backend implement which handles numerical computation for operators by memory management and instruction-level optimization, the geometry computation which manages data layout and memory mapping and is used for operators that change tensor shapes or memory arrangements, the combinator implementation which builds new operator functions by composing existing MNN operators.
- File name rule: given an operator named OP, then a CPU backend implementation must provide CPUOp.hpp and CPUOp.cpp files separately, if implemented as a combinator, provide OnnxOp.cpp file, if implemented via geometry computation, provide GeometryOp.cpp file.
- Code content rule: Each file name must appear as the very first line inside its corresponding code block. If provide cpu backend implement, organize the code content according to the given example. 
- Implement onResize and onExecute. In onResize, allocate the cache buffer using backend()->onAcquireBuffer(&mCache, Backend::DYNAMIC) and release it with backend()->onReleaseBuffer(&mCache, Backend::DYNAMIC), allowing the freed memory to be reused. In onExecute, perform necessary input validation to catch issues early. Return NO_ERROR upon successful execution.

"""
    prompt = USER_prompt_template

    # achievement = {
    #     "CPU": "Implement a custom CPU backend operator by providing CPUMyCustomOp.hpp_like and CPUMyCustomOp.cpp_like files",
    #     "Combiner": "Define a compositor that reuses and combines existing MNN operators by providing OnnxMyCustomOp.cpp_like file",
    #     "Geometry": "Use geometry computation to express the operator by providing GeometryMyCustomOp.cpp_like file",
    # }
    # problem_instruction = f"""Based on the following PyTorch model example, choose the most efficient approach to support the operator in the target model and implement computation acceleration. Only provide the file contents corresponding to your chosen implementation method, giving code but no any annotations."""
#
    # problem_instruction = f"""Based on the following PyTorch model example, {achievement[OP_type]} to achieve speedups. Only provide the file contents corresponding to your chosen implementation method, giving code but no any annotations."""


    if example_cpu_model != "" and example_cpu_MNN_OP != "":
        prompt += f"""Here are some basic example:

===== Example 1 Start =====

Given PyTorch model Det, using {backend} backend implementation:
``` \n
{example_cpu_model}
``` \n

You should respond with the final answer:
```
{example_cpu_MNN_OP}
``` \n
===== Example 1 End =====
    """
        
    if example_combiner_model != "" and example_combiner_MNN_OP != "":
        prompt += f"""
===== Example 2 Start =====
Given PyTorch model Gather, using Combiner implementation: \n
``` \n
{example_combiner_model}
``` \n

You should respond with the final answer:
```
    {example_combiner_MNN_OP}
``` \n
===== Example 2 End =====
        """
    if example_geo_model != "" and example_geo_MNN_OP != "":
        prompt += f"""
===== Example 3 Start =====

Given PyTorch model Det, using  Geometry implementation: \n
``` \n
{example_geo_model}
``` \n

You should respond with the final answer:
```
{example_geo_MNN_OP}
``` \n
===== Example 3 End =====
        """

    prompt += f"""
Now you are given the following PyTorch model:
``` \n
{targ_model}
```
    """
    achievement = {
        "CPU": "Implement a custom CPU backend operator by providing CPUMyCustomOp.hpp_like and CPUMyCustomOp.cpp_like files",
        "Combiner": "Define a compositor that reuses and combines existing MNN operators by providing OnnxMyCustomOp.cpp_like file",
        "Geometry": "Use geometry computation to express the operator by providing GeometryMyCustomOp.cpp_like file",
    }
    
    prompt += f" {achievement[OP_type]} and name operator with {op_name}. Only response with your final answer."
    return prompt

def gen_fewshot_prompt(target_model:str, 
                       target_model_type:str, 
                       compute_type:str,
                       op_name:str,
                       conv_type:str='Dense') -> str:
    model_type_list = ["normal","unary","binary","reduction","convolution"]
    compute_type_list = ["Atomic","Combiner","Geometry"]

    assert target_model_type in model_type_list, f"target_model_type should be one of {model_type_list}"
    assert compute_type in compute_type_list, f"compute_type should be one of {compute_type_list}"
    
    combiner = f"Implement a combiner for the corresponding operator in this PyTorch model by reusing and combining existing MNN operators. You need to write Onnx{op_name}.cpp file",
    normal_op = {
        "Atomic":f"Implement the CPU backend for the corresponding operator in this PyTorch model. You need to write the CPU{op_name}.hpp and CPU{op_name}.cpp files seperately",
        "Geometry":f"Implement the Geometry computing for the corresponding operator in this PyTorch model. You need to write the Geometry{op_name}.cpp file.",
        "Combiner":f"Implement a combiner for the corresponding operator in this PyTorch model by reusing and combining existing MNN operators. You need to write Onnx{op_name}.cpp file",
    }
    unary_op = {
        "Atomic":"Implement the CPU backend for the corresponding operator in this PyTorch model within the MNN framework, which belongs to one of MNN’s Unary types.  You are provided with the CPURUnary.hpp file and partial function implementations from CPUUnary.cpp. Based on these,  implement the complete CPUUnary.cpp file to enable accelerated computation of this PyTorch operator in the MNN framework.",
        "Geometry":"Implement the Geometry computing for the corresponding operator in MNN, which belongs to one of MNN’s geometry Unary types. In MNN, all  Unary operators’  geometry computing logic for is centrally handled by GeometryUnary.cpp. Write the GeometryUnary.cpp file to realize the geometry computing for this operator.",
        "Combiner":combiner,
    }
    binary_op = {
        "Atomic":"Implement the CPU backend for the corresponding operator in this PyTorch model within the MNN framework，which belongs to one of MNN’s Unary types.  You are provided with the CPURBinary.hpp file and partial function implementations from CPUBinary.cpp. Based on these,  implement the complete CPUBinary.cpp file to enable accelerated computation of this PyTorch operator in the MNN framework.",
        "Geometry":"Implement the Geometry computing for the corresponding operator in MNN, which belongs to one of MNN’s geometry Unary types. In MNN, all  Binary operators’  geometry computing logic for is centrally handled by GeometryBinary.cpp. Write the GeometryBinary.cpp file to realize the geometry computing for this operator.",
        "Combiner":combiner,
    }
    mask_reduction_extra_info = remove_reduction_class(REDUCTION_CPP, op_name) if target_model_type == "reduction" else ""
    reduction_op = {
        "Atomic":f"Implement the CPU backend for the corresponding operator in this PyTorch model within the MNN framework，which belongs to one of MNN’s Reduction types. In MNN, all Reduction operators are encapsulated by the abstract Reduction base class.To implement a specific reduction operation, you must inherit from this base class and override the onReduce method. You are provided with the CPUReduction.hpp file and partial function implementations from CPUReduction.cpp. Based on the extra information, implement the complete CPUReduction.cpp file to enable accelerated computation of this PyTorch operator in the MNN framework.",
        "Geometry":"Implement the Geometry computing for the corresponding operator in MNN, which belongs to one of MNN’s geometry Reduce types. In MNN, all  Reduce operators’  geometry computing logic for is centrally handled by GeometryReduce.cpp. Write the GeometryReduce.cpp file to realize the geometry computing for this operator.",
        "Combiner":combiner,
    }
    conv_content = {
        "Winograd":"ConvolutionPackWinograd.hpp",
        "Strassen":"Convolution1x1Strassen.hpp",
        "Dense":"DenseConvolutionTiledExecutor.hpp",
        "Group":"ConvolutionGroup.hpp",
    }
    conv_hpp_extra_info = CONV_EX_INFO[conv_type] if conv_type is not None else ""
    convolution_op = {
        "Atomic":f"Implement the CPU backend for the corresponding operator in this PyTorch model within the MNN framework. The operator is kind of {conv_type} convolution. You are provided the {conv_content[conv_type]} file, and write the {conv_content[conv_type].replace('.hpp', '.cpp')} file to enable accelerated computation of this PyTorch operator in the MNN framework.",
        "Geometry":"Implement the Geometry computing for the corresponding operator in MNN, which belongs to one of MNN’s geometry convolution types. In MNN, all convolution operators’  geometry computing logic for is centrally handled by GeometryConv2D.cpp. Write the GeometryConv2D.cpp file to realize the geometry computing for this operator.",
        "Combiner":combiner,
    }
    task_dic = {
        "normal": normal_op[compute_type],
        "unary": unary_op[compute_type],
        "binary": binary_op[compute_type],
        "reduction": reduction_op[compute_type],
        "convolution": convolution_op[compute_type],
    }

    extra_info_dic = {
        "normal": "",
        "unary": UNARY_EX_INFO,
        "binary": BINARY_EX_INFO,
        "reduction": mask_reduction_extra_info,
        "convolution": conv_hpp_extra_info,
    }
    extra_info = extra_info_dic[target_model_type] if compute_type == "Atomic" else ""
    task_description = task_dic[target_model_type]
    prompt = FEW_SHOT_PMPT.replace("__TARGET_MODEL__", target_model).replace("__EXTRA_INFO__", extra_info) + task_description

    return prompt

def remove_reduction_class(code: str, keyword: str) -> str:
    """
    delete Reduce class。
    case:
        keyword = "reduce_mean" → del MeanReduce
        keyword = "mn" → del MinReduce
        keyword = "maximumxxxxx" → del MaxReduce
        keyword = "pro" → del ProdReduce

    param:
        code:   C++ txt
        keyword: input
    """

    # reduce keywords
    reduce_keywords = {
        "MEAN": "MeanReduce",
        "SUM": "SumReduce",
        "MIN": "MinReduce",
        "MAX": "MaxReduce",
        "PROD": "ProdReduce",
        "ANY": "AnyReduce",
        "ALL": "AllReduce",
    }

    keyword_lower = keyword.lower()

    # 1. match reduce type
    matched_type = None
    for key in reduce_keywords.keys():
        if key.lower() in keyword_lower:
            matched_type = key
            break

    if matched_type is None:
        raise ValueError(
            f"None match: '{keyword}' "
        )

    # 2. get class name
    target_class = reduce_keywords[matched_type]

    # 3. del class <ClassName> : public Reduction { ... };
    pattern = (
        r"class\s+"
        + re.escape(target_class)
        + r"\s*:\s*public\s+Reduction\s*\{.*?\};"
    )

    new_code = re.sub(pattern, "", code, flags=re.DOTALL | re.IGNORECASE)
    return new_code

def generate_prompt_with_py_model(op_name,
                                  op_type,
                                  op_ctg,
                                  py_folder:Path=None, # path/to/py/folder
                                  save_folder:Path=None, # path/to/save/folder
                                  ):
    py_path = py_folder / op_type / op_ctg / (op_name+".py")
    py_model_contxt = read_file(py_path)
    prompt = gen_fewshot_prompt(
                            op_name=op_name,
                            target_model=py_model_contxt,
                            target_model_type=op_ctg,
                            compute_type=op_type,
                            conv_type=op_name.split("_")[0] if "_Convolution" in op_name \
                                                            else "Dense" # Dense/Winograd/Strassen/Group
                        )
    if save_folder is not None:
        save_path = save_folder / op_type / op_ctg /(op_name+".txt")
        with open(save_path,"w") as f:
            f.write(prompt)

    return prompt


def prepare_all_prompt(py_folder:Path,save_prompt_folder:Path):
    '''
    dataset_folder: root folder
    folder structure:
    dataset_folder/
        compute_type/ # Atomic/Combiner/Geometry
            target_model_type/ # normal/unary/binary/reduction/convolution
                pytorch_model.py
    '''

    pyfile_list =list_all_txt_or_py(py_folder)
    for file_path in pyfile_list:
        op_info = parse_op_info_from_path(file_path)
        pymodel_content = read_file(file_path)
        prompt = gen_fewshot_prompt(
                        target_model=pymodel_content,
                        target_model_type=op_info["op_ctg"],
                        compute_type=op_info["op_type"],
                        op_name=op_info["op_name"].split("_")[0],
                        conv_type=op_info["op_name"].split("_")[0] if "_Convolution" in op_info["op_name"] \
                                                                else "Dense" # Dense/Winograd/Strassen/Group
                    )
        print("--"*40)
        print(f"Operator file Name:{op_info["op_name"]}, {op_info["op_type"]}:{op_info["op_ctg"]}")
        print("--"*40)
        save_path = save_prompt_folder / op_info["op_type"] / op_info["op_ctg"]
        os.makedirs(save_path, exist_ok=True)
        save_file_path = save_path / (op_info["op_name"]+'.txt')
        with open(save_file_path, "w") as f:
            f.write(prompt)
        print(f"OpName:{op_info["op_name"]}. Saved prompt to {save_file_path}")
        print("--"*40)

if __name__ == "__main__":
    # test prompt generation
    pass



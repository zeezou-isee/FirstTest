//
//  CPUUnary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUUnary.hpp"
#include "UnaryUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "compute/ConvOpt.h"
#include "compute/CommonOptFunction.h"
#include <MNN/AutoTime.hpp>
#include "math/Vec.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
using VecType = Math::Vec<int8_t, 8>;
CPUUnary::CPUUnary(Backend *b, MNNUnaryExecute proc, MNNUnaryExecuteInt8 procInt8, const Op* op) : MNN::Execution(b), mProc(proc), mProcInt8(procInt8){
    if (op->main_as_UnaryOp()->tableInt8()) {
        mTableBuffer.resize(255);
        ::memcpy(mTableBuffer.data(), op->main_as_UnaryOp()->tableInt8()->data(), 255);
    }
}

ErrorCode CPUUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    MNN_ASSERT(inputs[0]->getType() == halide_type_of<float>() || inputs[0]->getType() == halide_type_of<int32_t>());
#ifdef MNN_SUPPORT_QUANT_EXTEND
    if (mProcInt8) {
        auto quantIn = TensorUtils::getDescribe(inputs[0])->quantAttr;
        auto quantOut = TensorUtils::getDescribe(outputs[0])->quantAttr;
        float outpScale = quantOut->scale == 0.f ? 0.f: 1.0f / quantOut->scale;
        mInpScale.push_back(quantIn->scale);
        mOupScale.push_back(outpScale);
        mInpZeroPoint.push_back(quantIn->zero);
        mOupZeroPoint.push_back(quantOut->zero);
        mMaxMinValue = {static_cast<ssize_t>(quantOut->min), static_cast<ssize_t>(quantOut->max)};
    }
#endif
    return NO_ERROR;
}

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

static MNNUnaryExecute selectForInt(int type) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return _unaryOp<UnaryAbs<int32_t>, int32_t>;
        case UnaryOpOperation_NEG:
            return _unaryOp<UnaryNeg<int32_t>, int32_t>;
        case UnaryOpOperation_SQUARE:
            return _unaryOp<UnarySquare<int32_t>, int32_t>;
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<int32_t>, int32_t>;
        default:
            break;
    }
    return nullptr;
}


MNNUnaryExecuteInt8 CPUUnary::selectForInt8(int type) {
#ifdef MNN_SUPPORT_QUANT_EXTEND
    switch (type) {
        case UnaryOpOperation_ABS:
            return _ABSInt8;
        case UnaryOpOperation_NEG:
            return _NegInt8;
        case UnaryOpOperation_SIGN:
            return _SignInt8;
        default:
            break;
    }
#endif
    return nullptr;
}
ErrorCode CPUUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto size = static_cast<CPUBackend*>(backend())->getTensorSize(input);
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(size);
    auto inputPtr = input->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    int outBytes = output->getType().bytes();
    if (halide_type_float == output->getType().code) {
        outBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    if (mTableBuffer.data()) {
#ifdef MNN_USE_SSE
        uint8_t* srcO = inputPtr;
        uint8_t* dstO = outputPtr;
        int offset = 128;
#else
        int8_t* srcO = (int8_t*)inputPtr;
        int8_t* dstO = (int8_t*)outputPtr;
        int offset = 0;
#endif
        MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
            int start = schedule.first * (int)tId;
            int realSize = schedule.first;
            if (tId == schedule.second -1 ) {
                realSize = size - start;
            }
            if (realSize > 0) {
                auto inp = srcO + start;
                auto out = dstO + start;
                for (int i = 0; i < realSize; ++i) {
                    int idx = inp[i] - offset + 127;
                    out[i] = offset + mTableBuffer[idx];
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    if (mProcInt8) {
        MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
            QuanPrePostParameters params;
            params.inputScale = mInpScale.data();
            params.outputScale = mOupScale.data();
            params.inputZeroPoint= mInpZeroPoint.data();
            params.outputZeroPoint = mOupZeroPoint.data();
            params.maxValue = mMaxMinValue[1];
            params.minValue = mMaxMinValue[0];
            int start = schedule.first * (int)tId;
            int realSize = schedule.first;
            if (tId == schedule.second -1 ) {
                realSize = size - start;
            }
            if (realSize > 0) {
                auto inp = inputPtr + start;
                auto out = outputPtr + start;
                mProcInt8(out, inp, realSize, &params);
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = size - start;
        }
        if (realSize > 0) {
            auto inp = inputPtr + start * outBytes;
            auto out = outputPtr + start * outBytes;
            mProc(out, inp, realSize);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUUnaryCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto core = static_cast<CPUBackend*>(backend)->functions();
        auto precision = static_cast<CPUBackend*>(backend)->precisionMode();
        auto type = inputs[0]->getType();
        MNNUnaryExecute proc = nullptr;
        MNNUnaryExecuteInt8 procInt8 = nullptr;
#ifdef MNN_SUPPORT_QUANT_EXTEND
        if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
            if (nullptr == core->MNNSelectUnaryFunctionForInt8) {
                return nullptr;
            }
            procInt8 = core->MNNSelectUnaryFunctionForInt8(op->main_as_UnaryOp()->opType());
        }
#endif
        if (nullptr == procInt8) {
            if (type.code == halide_type_int) {
                proc = selectForInt(op->main_as_UnaryOp()->opType());
            } else if (type.code == halide_type_float) {
                proc = core->MNNSelectUnaryFunctionForFloat(op->main_as_UnaryOp()->opType(), static_cast<CPUBackend*>(backend)->precisionMode());
               
            }
        }
        if (nullptr == proc && nullptr == procInt8 && nullptr == op->main_as_UnaryOp()->tableInt8()) {
            MNN_ERROR("ERROR: Unary Op can not execute\n");
            return nullptr;
        }
        return new CPUUnary(backend, proc, procInt8, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUUnaryCreator, OpType_UnaryOp);

} // namespace MNN

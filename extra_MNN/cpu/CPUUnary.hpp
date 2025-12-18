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


static void _Neg(void* out, const void* inp, int realSize) {
    MNNScaleAndAddBiasScalar((float*)out, (const float*)inp, 0.0f, -1.0f, realSize);
}

#ifdef MNN_SUPPORT_QUANT_EXTEND

#ifdef MNN_USE_NEON
static inline void exeNegInt8 (int8_t* out, const int8_t* inp, int sizeQuad, int8x8_t inZeroPoint, int8x8_t outZeroPoint, float32x4_t inpScale, float32x4_t outScale) {
    for (int i = 0;i < sizeQuad; ++i) {
        int8x16_t negValue = vld1q_s8(inp);
        int16x8_t val16_0 = vmovl_s8(vget_low_s8(negValue));
        int16x8_t val16_1 = vmovl_s8(vget_high_s8(negValue));
        val16_0 = vsubw_s8(val16_0, inZeroPoint);
        val16_1 = vsubw_s8(val16_1, inZeroPoint);
        int32x4_t val32_00 = vmovl_s16(vget_low_s16(val16_0));
        int32x4_t val32_01 = vmovl_s16(vget_high_s16(val16_0));
        int32x4_t val32_10 = vmovl_s16(vget_low_s16(val16_1));
        int32x4_t val32_11 = vmovl_s16(vget_high_s16(val16_1));
        float32x4_t valF_00 = vcvtq_f32_s32(val32_00);
        float32x4_t valF_01 = vcvtq_f32_s32(val32_01);
        float32x4_t valF_10 = vcvtq_f32_s32(val32_10);
        float32x4_t valF_11 = vcvtq_f32_s32(val32_11);
        valF_00 = vmulq_f32(valF_00, inpScale);
        valF_01 = vmulq_f32(valF_01, inpScale);
        valF_10 = vmulq_f32(valF_10, inpScale);
        valF_11 = vmulq_f32(valF_11, inpScale);
        valF_00 = vnegq_f32(valF_00);
        valF_01 = vnegq_f32(valF_01);
        valF_10 = vnegq_f32(valF_10);
        valF_11 = vnegq_f32(valF_11);
        valF_00 = vmulq_f32(valF_00, outScale);
        valF_01 = vmulq_f32(valF_01, outScale);
        valF_10 = vmulq_f32(valF_10, outScale);
        valF_11 = vmulq_f32(valF_11, outScale);
        int32x4_t val_00 = vcvtq_s32_f32(valF_00);
        int32x4_t val_01 = vcvtq_s32_f32(valF_01);
        int32x4_t val_10 = vcvtq_s32_f32(valF_10);
        int32x4_t val_11 = vcvtq_s32_f32(valF_11);
        int16x4_t v16_0 = vqmovn_s32(val_00);
        int16x4_t v16_1 = vqmovn_s32(val_01);
        int16x4_t v16_2 = vqmovn_s32(val_10);
        int16x4_t v16_3 = vqmovn_s32(val_11);
        int16x8_t v16_4 = vcombine_s16(v16_0, v16_1);
        int16x8_t v16_5 = vcombine_s16(v16_2, v16_3);
        v16_4 = vaddw_s8(v16_4, outZeroPoint);
        v16_5 = vaddw_s8(v16_5, outZeroPoint);
        int8x8_t v8_0 = vqmovn_s16(v16_4);
        int8x8_t v8_1 = vqmovn_s16(v16_5);

        vst1_s8(out, v8_0);
        vst1_s8(out + 8, v8_1);
        inp  += 16;
        out += 16;
    }
}
#endif
static void _NegInt8(void* out, const void* inp, int realSize, QuanPrePostParameters* params) {
    int sizeDiv16 = realSize / 16;
    int remain = realSize % 16;
#ifdef MNN_USE_NEON
    int8_t* outPtr = (int8_t*)out;
    int8_t* inPtr  = (int8_t*)inp;
    int8x8_t inZeroPoint = vdup_n_s8(params->inputZeroPoint[0]);
    int8x8_t outZeroPoint = vdup_n_s8(params->outputZeroPoint[0]);
    float32x4_t inpScale = vdupq_n_f32(params->inputScale[0]);
    float32x4_t outScale = vdupq_n_f32(params->outputScale[0]);
    if (sizeDiv16 > 0) {
        exeNegInt8(outPtr, inPtr, sizeDiv16, inZeroPoint, outZeroPoint, inpScale, outScale);
    }
    if (remain > 0) {
        int8_t intmp[16] = {0};
        int8_t outmp[16] = {0};
        ::memcpy(intmp, reinterpret_cast<const int8_t*>(inp) + 16 * sizeDiv16, remain * sizeof(int8_t));
        exeNegInt8(outmp, intmp, 1, inZeroPoint, outZeroPoint, inpScale, outScale);
        ::memcpy(reinterpret_cast<int8_t*>(out) + 16 * sizeDiv16, outmp, remain * sizeof(int8_t));
    }
#else
#ifdef MNN_USE_SSE
    uint8_t* dst = (uint8_t*)out;
    uint8_t* src = (uint8_t*)inp;
    int offset = 128;
#else
    int8_t* dst = (int8_t*)out;
    int8_t* src = (int8_t*)inp;
    int offset = 0;
#endif
    int inzero_     = static_cast<int>(params->inputZeroPoint[0]);
    int outzero_    = static_cast<int>(params->outputZeroPoint[0]);
    float inscale_  = params->inputScale[0];
    float outscale_ = params->outputScale[0];
    int min_        = static_cast<int>(params->minValue);
    int max_        = static_cast<int>(params->maxValue);
    for (int i = 0; i < realSize; ++i) {
        int value = -(src[i] - inzero_ - offset) * inscale_ * outscale_ + outzero_;
        if (value > max_) {
            value = max_;
        }
        if (value < min_) {
            value = min_;
        }
        dst[i] = value + offset;
    }
#endif
}

#ifdef MNN_USE_NEON
static inline void exeAbsInt8(int8_t* out, const int8_t* inp, int sizeQuad, int8x8_t inZeroPoint, int8x8_t outZeroPoint, float32x4_t inpScale, float32x4_t outScale) {
    for (int i = 0;i < sizeQuad; ++i) {
        int8x16_t absValue = vld1q_s8(inp);
        int16x8_t val16_0 = vmovl_s8(vget_low_s8(absValue));
        int16x8_t val16_1 = vmovl_s8(vget_high_s8(absValue));
        val16_0 = vsubw_s8(val16_0, inZeroPoint);
        val16_1 = vsubw_s8(val16_1, inZeroPoint);
        int32x4_t val32_00 = vmovl_s16(vget_low_s16(val16_0));
        int32x4_t val32_01 = vmovl_s16(vget_high_s16(val16_0));
        int32x4_t val32_10 = vmovl_s16(vget_low_s16(val16_1));
        int32x4_t val32_11 = vmovl_s16(vget_high_s16(val16_1));
        float32x4_t valF_00 = vcvtq_f32_s32(val32_00);
        float32x4_t valF_01 = vcvtq_f32_s32(val32_01);
        float32x4_t valF_10 = vcvtq_f32_s32(val32_10);
        float32x4_t valF_11 = vcvtq_f32_s32(val32_11);
        valF_00 = vmulq_f32(valF_00, inpScale);
        valF_01 = vmulq_f32(valF_01, inpScale);
        valF_10 = vmulq_f32(valF_10, inpScale);
        valF_11 = vmulq_f32(valF_11, inpScale);
        valF_00 = vabsq_f32(valF_00);
        valF_01 = vabsq_f32(valF_01);
        valF_10 = vabsq_f32(valF_10);
        valF_11 = vabsq_f32(valF_11);
        valF_00 = vmulq_f32(valF_00, outScale);
        valF_01 = vmulq_f32(valF_01, outScale);
        valF_10 = vmulq_f32(valF_10, outScale);
        valF_11 = vmulq_f32(valF_11, outScale);
        int32x4_t val_00 = vcvtq_s32_f32(valF_00);
        int32x4_t val_01 = vcvtq_s32_f32(valF_01);
        int32x4_t val_10 = vcvtq_s32_f32(valF_10);
        int32x4_t val_11 = vcvtq_s32_f32(valF_11);
        int16x4_t v16_0 = vqmovn_s32(val_00);
        int16x4_t v16_1 = vqmovn_s32(val_01);
        int16x4_t v16_2 = vqmovn_s32(val_10);
        int16x4_t v16_3 = vqmovn_s32(val_11);
        int16x8_t v16_4 = vcombine_s16(v16_0, v16_1);
        int16x8_t v16_5 = vcombine_s16(v16_2, v16_3);
        v16_4 = vaddw_s8(v16_4, outZeroPoint);
        v16_5 = vaddw_s8(v16_5, outZeroPoint);
        int8x8_t v8_0 = vqmovn_s16(v16_4);
        int8x8_t v8_1 = vqmovn_s16(v16_5);

        vst1_s8(out, v8_0);
        vst1_s8(out + 8, v8_1);
        inp  += 16;
        out += 16;
    }
}
#endif
static void _ABSInt8(void* out, const void* inp, int realSize, QuanPrePostParameters* params) {
    int sizeDiv16 = realSize / 16;
    int remain = realSize % 16;
#ifdef MNN_USE_NEON
    int8_t* outPtr = (int8_t*)out;
    int8_t* inPtr  = (int8_t*)inp;
    int8x8_t inZeroPoint = vdup_n_s8(params->inputZeroPoint[0]);
    int8x8_t outZeroPoint = vdup_n_s8(params->outputZeroPoint[0]);
    float32x4_t inpScale = vdupq_n_f32(params->inputScale[0]);
    float32x4_t outScale = vdupq_n_f32(params->outputScale[0]);
    if (sizeDiv16 > 0) {
        exeAbsInt8(outPtr, inPtr, sizeDiv16, inZeroPoint, outZeroPoint, inpScale, outScale);
    }
    if (remain > 0) {
        int8_t intmp[16] = {0};
        int8_t outmp[16] = {0};
        ::memcpy(intmp, reinterpret_cast<const int8_t*>(inp) + 16 * sizeDiv16, remain * sizeof(int8_t));
        exeAbsInt8(outmp, intmp, 1, inZeroPoint, outZeroPoint, inpScale, outScale);
        ::memcpy(reinterpret_cast<int8_t*>(out) + 16 * sizeDiv16, outmp, remain * sizeof(int8_t));
    }
#else
#ifdef MNN_USE_SSE
    uint8_t* dst = (uint8_t*)out;
    uint8_t* src = (uint8_t*)inp;
    int offset = 128;
#else
    int8_t* dst = (int8_t*)out;
    int8_t* src = (int8_t*)inp;
    int offset = 0;
#endif
    int inzero_  = static_cast<int>(params->inputZeroPoint[0]);
    int outzero_ = static_cast<int>(params->outputZeroPoint[0]);
    for (int i = 0; i < realSize; ++i) {
        auto value = abs((src[i] - inzero_ - offset) * params->inputScale[0]);
        value = value * params->outputScale[0] + outzero_;
        if (value > params->maxValue) {
            value = params->maxValue;
        }
        if (value < params->minValue) {
            value = params->minValue;
        }
        dst[i] = value + offset;
    }
#endif
}
#ifdef MNN_USE_NEON
static inline void exeSignInt8 (int8_t* out, const int8_t* inp, int sizeQuad, int16x8_t one, int16x8_t negone, int16x8_t zero, int8x8_t inZeroPoint, int8x8_t outZeroPoint, float32x4_t outScale) {
        for (int i = 0;i < sizeQuad; ++i) {
            int8x16_t value = vld1q_s8(inp);
            int16x8_t vallow = vmovl_s8(vget_low_s8(value));
            int16x8_t valhi = vmovl_s8(vget_high_s8(value));
            vallow = vsubw_s8(vallow, inZeroPoint);
            valhi  = vsubw_s8(valhi, inZeroPoint);
            uint16x8_t lomask1  = vcgtq_s16(vallow, zero);
            uint16x8_t lomask_1 = vcltq_s16(vallow, zero);
            uint16x8_t himask1  = vcgtq_s16(valhi, zero);
            uint16x8_t himask_1 = vcltq_s16(valhi, zero);
            uint16x8_t zeromask_low = vceqq_u16(lomask1, lomask_1);
            uint16x8_t zeromask_hi = vceqq_u16(himask1, himask_1);
            vallow = vbslq_s16(lomask1, one, negone);
            vallow = vbslq_s16(zeromask_low, zero, vallow);
            valhi = vbslq_s16(himask1, one, negone);
            valhi = vbslq_s16(zeromask_hi, zero, valhi);
            int8x8_t v8_0 = vqmovn_s16(vallow);
            int8x8_t v8_1 = vqmovn_s16(valhi);
            vst1_s8(out, v8_0);
            vst1_s8(out + 8, v8_1);
            inp  += 16;
            out += 16;
        }
}
#endif
static void _SignInt8(void* out, const void* inp, int realSize, QuanPrePostParameters* params) {
    int sizeDiv16 = realSize / 16;
    int remain = realSize % 16;
#ifdef MNN_USE_NEON
    int8_t* outPtr = (int8_t*)out;
    int8_t* inPtr  = (int8_t*)inp;
    int16x8_t one = vdupq_n_s16(1);
    int16x8_t negone = vdupq_n_s16(-1);
    int16x8_t zero = vdupq_n_s16(0);
    int8x8_t inZeroPoint = vdup_n_s8(params->inputZeroPoint[0]);
    int8x8_t outZeroPoint = vdup_n_s8(params->outputZeroPoint[0]);
    float32x4_t outScale = vdupq_n_f32(params->outputScale[0]);
    if (sizeDiv16 > 0) {
        exeSignInt8(outPtr, inPtr, sizeDiv16, one, negone, zero, inZeroPoint, outZeroPoint, outScale);
    }
    if (remain > 0) {
        int8_t intmp[16] = {0};
        int8_t outmp[16] = {0};
        ::memcpy(intmp, reinterpret_cast<const int8_t*>(inp) + 16 * sizeDiv16, remain * sizeof(int8_t));
        exeSignInt8(outmp, intmp, 1, one, negone, zero, inZeroPoint, outZeroPoint, outScale);
        ::memcpy(reinterpret_cast<int8_t*>(out) + 16 * sizeDiv16, outmp, remain * sizeof(int8_t));
    }
#else
#ifdef MNN_USE_SSE
    uint8_t* dst = (uint8_t*)out;
    uint8_t* src = (uint8_t*)inp;
    int offset = 128;
#else
    int8_t* dst = (int8_t*)out;
    int8_t* src = (int8_t*)inp;
    int offset = 0;
#endif
    int inzero_  = static_cast<int>(params->inputZeroPoint[0]);
    int outzero_ = static_cast<int>(params->outputZeroPoint[0]);
    for (int i = 0; i < realSize; ++i) {
        auto value = src[i] - offset - inzero_;
        if (value > 0) {
            int f = 1 * params->outputScale[0] + outzero_;
            dst[i]     = f + offset;
        } else if (value < 0) {
            int f = -1 * params->outputScale[0] + outzero_;
            dst[i]     = f + offset;
        } else {
            dst[i] = outzero_ + offset;
        }
    }
#endif
}
#endif
static void _ABS(void* out, const void* inp, int realSize) {
    MNNReluWithSlopeCommon((float*)out, (const float*)inp, realSize, -1.0f);
}
static void _Square(void* out, const void* inp, int realSize) {
    MNNMatrixProdCommon((float*)out, (const float*)inp, (const float*)inp, realSize, 0, 0, 0, 1);
}

static void _EXP(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (float*)outRaw;
    auto inp = (const float*)inpRaw;
    float offset[] = {
        1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(out, inp, offset, realSize);
}
static void _EXPM1(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (float*)outRaw;
    auto inp = (const float*)inpRaw;
    float offset[] = {
        1.0f,
        -1.0f,
        0.0f,
        0.0f
    };
    MNNExp(out, inp, offset, realSize);
}

} // namespace MNN
#endif /* CPUUnary_hpp */

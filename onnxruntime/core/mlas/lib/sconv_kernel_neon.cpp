/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_kernel_neon.cpp

Abstract:

    This module implements the single precision convolution kernels for ARM NEON.

--*/

#include "sconv.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <algorithm>
#include <cstddef>

#include "mlasi.h"

// Common implementation for NCHW and NCHWC convolution kernels
template <bool IsNchwcFormat>
void
    MLASCALL
    MlasConvFloatKernelNeonImpl(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    (void)InputStride;

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {
            const float* filter = Filter + filterSetBlock * FilterStrideElements;
            float* output = Output + filterSetBlock * OutputStrideElements;

            float32x4_t Accumulator;

            if (AccumulateOutput) {
                Accumulator = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
            } else {
                Accumulator = MlasBroadcastFloat32x4(0.0f);
            }

            if (BiasAddition) {
                const float32x4_t BiasVector = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize]);
                Accumulator = MlasAddFloat32x4(Accumulator, BiasVector);
            }

            for (size_t kh = 0; kh < KernelHeight; kh++) {
                for (size_t kw = 0; kw < KernelWidth; kw++) {
                    const float* input_base = Input + output_idx * StrideWidthElements +
                                              kh * DilatedInputWidthElements + kw * DilationWidthElements;

                    if (IsNchwcFormat) {
                        // NCHWC format - process each element in the block
                        for (size_t filterBlock = 0; filterBlock < BlockSize; filterBlock++) {
                            const float* input_element = input_base + filterBlock;
                            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                            const float* input_row_end = input_row_start + InputWidthElements;

                            float input_value;
                            if (is_main_region || (input_element >= input_row_start && input_element < input_row_end)) {
                                input_value = *input_element;
                            } else {
                                input_value = 0.0f;
                            }

                            const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                            size_t kernel_base_pos = kh * (KernelWidth * BlockSize * BlockSize) +
                                                     kw * (BlockSize * BlockSize) +
                                                     filterBlock * BlockSize;

                            const float32x4_t FilterVector = MlasLoadFloat32x4(&filter[kernel_base_pos]);

                            Accumulator = MlasMultiplyAddFloat32x4(InputVector, FilterVector, Accumulator);
                        }
                    } else {
                        // NCHW format - simpler processing
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        float input_value;
                        if (is_main_region || (input_base >= input_row_start && input_base < input_row_end)) {
                            input_value = *input_base;
                        } else {
                            input_value = 0.0f;
                        }

                        const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                        size_t kernel_base_pos = kh * KernelWidth + kw;

                        const float32x4_t FilterVector = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize]);

                        Accumulator = MlasMultiplyAddFloat32x4(InputVector, FilterVector, Accumulator);
                    }
                }
            }

            if (ReluActivation) {
                Accumulator = MlasMaximumFloat32x4(Accumulator, ZeroVector);
            }

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator);
        }
    }
}

void
    MLASCALL
    MlasConvNchwFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    MlasConvFloatKernelNeonImpl<false>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
}

//
// Implementation of MlasConvNchwcFloatKernelNeon
//

void
    MLASCALL
    MlasConvNchwcFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    MlasConvFloatKernelNeonImpl<true>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
}

//
// Implementation of MlasConvDepthwiseFloatKernelNeon
//
// This kernel performs depthwise separable convolution where each input channel
// is convolved with its own filter. This is more efficient than standard convolution
// for certain network architectures like MobileNets.
//

void
    MLASCALL
    MlasConvDepthwiseFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    (void)InputStrideElements;

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    // Process outputs in pairs when possible for better data reuse
    size_t output_idx;
    for (output_idx = 0; output_idx + 1 < TotalOutputCount; output_idx += 2) {
        bool is_main_region0 = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);
        bool is_main_region1 = (output_idx + 1 >= OutputCountLeftPad && output_idx + 1 < OutputCountLeftPad + OutputCount);

        float32x4_t Accumulator0, Accumulator1;

        // Initialize accumulators for both positions
        if (AccumulateOutput) {
            Accumulator0 = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
            Accumulator1 = MlasLoadFloat32x4(&Output[(output_idx + 1) * BlockSize]);
        } else {
            Accumulator0 = MlasBroadcastFloat32x4(0.0f);
            Accumulator1 = MlasBroadcastFloat32x4(0.0f);
        }
        
        // Preserve the bias addition formula
        if (BiasAddition) {
            const float32x4_t BiasVector = MlasLoadFloat32x4(Bias);
            Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector);
            Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector);
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
            const float* input_row_end = input_row_start + InputWidthElements;
            
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                size_t kernel_pos = kh * KernelWidth + kw;
                const float32x4_t FilterVector = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);

                const float* input_base0 = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;
                const float* input_base1 = Input + (output_idx + 1) * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                float32x4_t InputVector0, InputVector1;

                // Handle first output position
                if (is_main_region0) {
                    InputVector0 = MlasLoadFloat32x4(input_base0);
                } else {
                    // Efficiently handle boundary conditions for first position
                    float input_values0[4];
                    
                    input_values0[0] = (input_base0 >= input_row_start && input_base0 < input_row_end) ? input_base0[0] : 0.0f;
                    input_values0[1] = (input_base0 + 1 >= input_row_start && input_base0 + 1 < input_row_end) ? input_base0[1] : 0.0f;
                    input_values0[2] = (input_base0 + 2 >= input_row_start && input_base0 + 2 < input_row_end) ? input_base0[2] : 0.0f;
                    input_values0[3] = (input_base0 + 3 >= input_row_start && input_base0 + 3 < input_row_end) ? input_base0[3] : 0.0f;
                    
                    InputVector0 = MlasLoadFloat32x4(input_values0);
                }

                // Handle second output position
                if (is_main_region1) {
                    InputVector1 = MlasLoadFloat32x4(input_base1);
                } else {
                    // Efficiently handle boundary conditions for second position
                    float input_values1[4];
                    
                    input_values1[0] = (input_base1 >= input_row_start && input_base1 < input_row_end) ? input_base1[0] : 0.0f;
                    input_values1[1] = (input_base1 + 1 >= input_row_start && input_base1 + 1 < input_row_end) ? input_base1[1] : 0.0f;
                    input_values1[2] = (input_base1 + 2 >= input_row_start && input_base1 + 2 < input_row_end) ? input_base1[2] : 0.0f;
                    input_values1[3] = (input_base1 + 3 >= input_row_start && input_base1 + 3 < input_row_end) ? input_base1[3] : 0.0f;
                    
                    InputVector1 = MlasLoadFloat32x4(input_values1);
                }

                // Process both positions with the same filter
                Accumulator0 = MlasMultiplyAddFloat32x4(InputVector0, FilterVector, Accumulator0);
                Accumulator1 = MlasMultiplyAddFloat32x4(InputVector1, FilterVector, Accumulator1);
            }
        }

        // Apply ReLU and store results
        if (ReluActivation) {
            Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
            Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], Accumulator0);
        MlasStoreFloat32x4(&Output[(output_idx + 1) * BlockSize], Accumulator1);
    }

    // Handle remaining single output position if TotalOutputCount is odd
    if (output_idx < TotalOutputCount) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        float32x4_t Accumulator;

        if (AccumulateOutput) {
            Accumulator = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
        } else {
            Accumulator = MlasBroadcastFloat32x4(0.0f);
        }
        
        // Preserve the bias addition formula
        if (BiasAddition) {
            Accumulator = MlasAddFloat32x4(Accumulator, MlasLoadFloat32x4(Bias));
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
            const float* input_row_end = input_row_start + InputWidthElements;
            
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                size_t kernel_pos = kh * KernelWidth + kw;

                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                float32x4_t InputVector;

                if (is_main_region) {
                    InputVector = MlasLoadFloat32x4(input_base);
                } else {
                    // Efficiently handle boundary conditions
                    float input_values[4];
                    
                    input_values[0] = (input_base >= input_row_start && input_base < input_row_end) ? input_base[0] : 0.0f;
                    input_values[1] = (input_base + 1 >= input_row_start && input_base + 1 < input_row_end) ? input_base[1] : 0.0f;
                    input_values[2] = (input_base + 2 >= input_row_start && input_base + 2 < input_row_end) ? input_base[2] : 0.0f;
                    input_values[3] = (input_base + 3 >= input_row_start && input_base + 3 < input_row_end) ? input_base[3] : 0.0f;
                    
                    InputVector = MlasLoadFloat32x4(input_values);
                }

                const float32x4_t FilterVector = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);

                Accumulator = MlasMultiplyAddFloat32x4(InputVector, FilterVector, Accumulator);
            }
        }

        if (ReluActivation) {
            Accumulator = MlasMaximumFloat32x4(Accumulator, ZeroVector);
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], Accumulator);
    }
}

//
// Implementation of MlasConvPointwiseFloatKernelNeon
//
// This kernel performs pointwise (1x1) convolution which is essentially
// a matrix multiplication across the channel dimension. It's optimized
// for cases where the kernel size is 1x1.
//

void
    MLASCALL
    MlasConvPointwiseFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t InputChannels,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t OutputCount,
        const float* Bias,
        unsigned KernelFlags
    )
{
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    for (size_t f = 0; f < FilterCount; f++) {
        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;
        const float* bias_ptr = BiasAddition ? &Bias[f * BlockSize] : nullptr;
        
        // Process outputs in pairs when possible for better data reuse
        size_t i;
        for (i = 0; i + 1 < OutputCount; i += 2) {
            // Process two output positions together
            float32x4_t Accumulator0, Accumulator1;
            
            // Initialize accumulators for both positions
            if (AccumulateOutput) {
                Accumulator0 = MlasLoadFloat32x4(&output[i * BlockSize]);
                Accumulator1 = MlasLoadFloat32x4(&output[(i + 1) * BlockSize]);
            } else {
                Accumulator0 = MlasBroadcastFloat32x4(0.0f);
                Accumulator1 = MlasBroadcastFloat32x4(0.0f);
            }
            
            if (BiasAddition) {
                const float32x4_t BiasVector = MlasLoadFloat32x4(bias_ptr);
                Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector);
                Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector);
            }
            
            for (size_t c = 0; c < InputChannels; c++) {
                const float* input_ptr0 = Input + c * InputStrideElements + i * StrideWidthElements;
                const float* input_ptr1 = Input + c * InputStrideElements + (i + 1) * StrideWidthElements;
                const float* filter_base = filter + c * BlockSize * BlockSize;
                
                // Load all 4 input values at once for each position
                const float32x4_t input_vec0 = MlasLoadFloat32x4(input_ptr0);
                const float32x4_t input_vec1 = MlasLoadFloat32x4(input_ptr1);
                
                // Load all filter values for this channel
                const float32x4_t FilterVector0 = MlasLoadFloat32x4(filter_base);
                const float32x4_t FilterVector1 = MlasLoadFloat32x4(filter_base + BlockSize);
                const float32x4_t FilterVector2 = MlasLoadFloat32x4(filter_base + BlockSize * 2);
                const float32x4_t FilterVector3 = MlasLoadFloat32x4(filter_base + BlockSize * 3);
                
                // Process first position - use vdupq_laneq_f32 to broadcast individual elements
                Accumulator0 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec0, 0), FilterVector0, Accumulator0);
                Accumulator0 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec0, 1), FilterVector1, Accumulator0);
                Accumulator0 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec0, 2), FilterVector2, Accumulator0);
                Accumulator0 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec0, 3), FilterVector3, Accumulator0);
                
                // Process second position
                Accumulator1 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec1, 0), FilterVector0, Accumulator1);
                Accumulator1 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec1, 1), FilterVector1, Accumulator1);
                Accumulator1 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec1, 2), FilterVector2, Accumulator1);
                Accumulator1 = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec1, 3), FilterVector3, Accumulator1);
            }
            
            // Apply ReLU and store results
            if (ReluActivation) {
                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
            }
            MlasStoreFloat32x4(&output[i * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[(i + 1) * BlockSize], Accumulator1);
        }
        
        // Handle remaining single output position if OutputCount is odd
        if (i < OutputCount) {
            float32x4_t Accumulator;
            if (AccumulateOutput) {
                Accumulator = MlasLoadFloat32x4(&output[i * BlockSize]);
            } else {
                Accumulator = MlasBroadcastFloat32x4(0.0f);
            }
            
            if (BiasAddition) {
                const float32x4_t BiasVector = MlasLoadFloat32x4(bias_ptr);
                Accumulator = MlasAddFloat32x4(Accumulator, BiasVector);
            }
            
            for (size_t c = 0; c < InputChannels; c++) {
                const float* input_ptr = Input + c * InputStrideElements + i * StrideWidthElements;
                const float* filter_base = filter + c * BlockSize * BlockSize;
                
                // Load all 4 input values at once
                const float32x4_t input_vec = MlasLoadFloat32x4(input_ptr);
                
                // Load all filter values for this channel
                const float32x4_t FilterVector0 = MlasLoadFloat32x4(filter_base);
                const float32x4_t FilterVector1 = MlasLoadFloat32x4(filter_base + BlockSize);
                const float32x4_t FilterVector2 = MlasLoadFloat32x4(filter_base + BlockSize * 2);
                const float32x4_t FilterVector3 = MlasLoadFloat32x4(filter_base + BlockSize * 3);
                
                // Process using vdupq_laneq_f32 to broadcast individual elements
                Accumulator = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec, 0), FilterVector0, Accumulator);
                Accumulator = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec, 1), FilterVector1, Accumulator);
                Accumulator = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec, 2), FilterVector2, Accumulator);
                Accumulator = MlasMultiplyAddFloat32x4(vdupq_laneq_f32(input_vec, 3), FilterVector3, Accumulator);
            }
            
            if (ReluActivation) {
                Accumulator = MlasMaximumFloat32x4(Accumulator, ZeroVector);
            }
            MlasStoreFloat32x4(&output[i * BlockSize], Accumulator);
        }
    }
}

#endif  // __aarch64__ || _M_ARM64
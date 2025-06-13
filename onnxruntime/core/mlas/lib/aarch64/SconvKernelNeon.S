/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvKernelNeon.S

Abstract:

    This module implements the kernels for the single precision convolution
    operation using ARM NEON instructions.

--*/

#include "asmmacro.h"
#include "SconvKernelCommon.h"

        .text

/*++

Macro Description:

    This macro generates code to clear the block accumulators.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    v16-v31 - Supplies the block accumulators.

--*/

        .macro ClearBlock FilterCount, OutputCount

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "eor v16.16b,v16.16b,v16.16b"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "eor v17.16b,v17.16b,v17.16b"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "eor v18.16b,v18.16b,v18.16b"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "eor v19.16b,v19.16b,v19.16b"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "eor v20.16b,v20.16b,v20.16b"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "eor v21.16b,v21.16b,v21.16b"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "eor v22.16b,v22.16b,v22.16b"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "eor v23.16b,v23.16b,v23.16b"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "eor v24.16b,v24.16b,v24.16b"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "eor v25.16b,v25.16b,v25.16b"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "eor v26.16b,v26.16b,v26.16b"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "eor v27.16b,v27.16b,v27.16b"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "eor v28.16b,v28.16b,v28.16b"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "eor v29.16b,v29.16b,v29.16b"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "eor v30.16b,v30.16b,v30.16b"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "eor v31.16b,v31.16b,v31.16b"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "eor v0.16b,v0.16b,v0.16b"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "eor v1.16b,v1.16b,v1.16b"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "eor v2.16b,v2.16b,v2.16b"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "eor v3.16b,v3.16b,v3.16b"

        .endm
/*++

Macro Description:

    This macro multiplies and accumulates for FilterCount by OutputCount block
    of the output buffer.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

    VectorOffset - Supplies the byte offset from the filter buffer to fetch
        elements.

    BroadcastOffset - Supplies the byte offset from the input buffer to fetch
        elements.

Implicit Arguments:

    x6 - Supplies the address of the input buffer.

    x7 - Supplies the address of the filter buffer.

    x1 - Supplies the FilterStride parameter (see function description).

    x15 - Supplies the address of the filter buffer plus 2 * FilterStride.

    x4 - Supplies the StrideWidth parameter (see function description).

    x14 - Supplies the address of the input buffer plus 3 * StrideWidth.

    v16-v31 - Supplies the block accumulators.

--*/

        .macro ComputeBlock KernelType, FilterCount, OutputCount, VectorOffset, BroadcastOffset

.ifeqs "\KernelType\()","Depthwise"
        ldr     q8,[x7,#\VectorOffset\()]
        EmitIfCountGE \OutputCount\(), 1, "add x13,x6,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 1, "ldr q9,[x13]"
        EmitIfCountGE \OutputCount\(), 1, "fmla v16.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x6,x4"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 2, "ldr q9,[x13]"
        EmitIfCountGE \OutputCount\(), 2, "fmla v17.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x6,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 3, "ldr q9,[x13]"
        EmitIfCountGE \OutputCount\(), 3, "fmla v18.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 4, "add x13,x14,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 4, "ldr q9,[x13]"
        EmitIfCountGE \OutputCount\(), 4, "fmla v19.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x14,x4"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 5, "ldr q9,[x13]"
        EmitIfCountGE \OutputCount\(), 5, "fmla v20.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x14,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 6, "ldr q9,[x13]"
        EmitIfCountGE \OutputCount\(), 6, "fmla v21.4s,v8.4s,v9.4s"
.else
.if \FilterCount\() == 1
        ldr     q8,[x7,#\VectorOffset\()]
        EmitIfCountGE \OutputCount\(), 1, "add x13,x6,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 1, "ld1r {v9.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 1, "fmla v16.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x6,x4"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 2, "ld1r {v9.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 2, "fmla v17.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x6,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 3, "ld1r {v9.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 3, "fmla v18.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 4, "add x13,x14,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 4, "ld1r {v9.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 4, "fmla v19.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x14,x4"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 5, "ld1r {v9.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 5, "fmla v20.4s,v8.4s,v9.4s"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x14,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 6, "ld1r {v9.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 6, "fmla v21.4s,v8.4s,v9.4s"
.else
        EmitIfCountGE \OutputCount\(), 1, "add x13,x6,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 1, "ld1r {v10.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x6,x4"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 2, "ld1r {v11.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x6,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 3, "ld1r {v12.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 4, "add x13,x14,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 4, "ld1r {v13.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x14,x4"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 5, "ld1r {v14.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x14,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 6, "ld1r {v15.4s},[x13]"
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "ldr q8,[x7,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 1, "fmla v16.4s,v8.4s,v10.4s"
        EmitIfCountGE \FilterCount\(), 2, "add x13,x7,x1"
        EmitIfCountGE \FilterCount\(), 2, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 2, "fmla v22.4s,v8.4s,v10.4s"
        EmitIfCountGE \FilterCount\(), 3, "ldr q8,[x15,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 3, "fmla v28.4s,v8.4s,v10.4s"
        EmitIfCountGE \FilterCount\(), 4, "add x13,x15,x1"
        EmitIfCountGE \FilterCount\(), 4, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 4, "fmla v0.4s,v8.4s,v10.4s"
.else
        EmitIfCountGE \FilterCount\(), 1, "ldr q8,[x7,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "fmla v16.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "fmla v17.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "fmla v18.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "fmla v19.4s,v8.4s,v13.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "fmla v20.4s,v8.4s,v14.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "fmla v21.4s,v8.4s,v15.4s"
        EmitIfCountGE \FilterCount\(), 2, "add x13,x7,x1"
        EmitIfCountGE \FilterCount\(), 2, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "fmla v22.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "fmla v23.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "fmla v24.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "fmla v25.4s,v8.4s,v13.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "fmla v26.4s,v8.4s,v14.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "fmla v27.4s,v8.4s,v15.4s"
        EmitIfCountGE \FilterCount\(), 3, "ldr q8,[x15,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "fmla v28.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "fmla v29.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "fmla v30.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "fmla v31.4s,v8.4s,v13.4s"
        EmitIfCountGE \FilterCount\(), 4, "add x13,x15,x1"
        EmitIfCountGE \FilterCount\(), 4, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "fmla v0.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "fmla v1.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "fmla v2.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "fmla v3.4s,v8.4s,v13.4s"
.endif
.endif
.ifeqs "\KernelType\()","Pointwise"
        EmitIfCountGE \OutputCount\(), 1, "add x13,x6,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 1, "ld1r {v10.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x6,x4"
        EmitIfCountGE \OutputCount\(), 2, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 2, "ld1r {v11.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x6,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 3, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 3, "ld1r {v12.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 4, "add x13,x14,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 4, "ld1r {v13.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x14,x4"
        EmitIfCountGE \OutputCount\(), 5, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 5, "ld1r {v14.4s},[x13]"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x14,x4,lsl #1"
        EmitIfCountGE \OutputCount\(), 6, "add x13,x13,#\BroadcastOffset\()"
        EmitIfCountGE \OutputCount\(), 6, "ld1r {v15.4s},[x13]"
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "ldr q8,[x7,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 1, "fmla v16.4s,v8.4s,v10.4s"
        EmitIfCountGE \FilterCount\(), 2, "add x13,x7,x1"
        EmitIfCountGE \FilterCount\(), 2, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 2, "fmla v22.4s,v8.4s,v10.4s"
        EmitIfCountGE \FilterCount\(), 3, "ldr q8,[x15,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 3, "fmla v28.4s,v8.4s,v10.4s"
        EmitIfCountGE \FilterCount\(), 4, "add x13,x15,x1"
        EmitIfCountGE \FilterCount\(), 4, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCountGE \FilterCount\(), 4, "fmla v0.4s,v8.4s,v10.4s"
.else
        EmitIfCountGE \FilterCount\(), 1, "ldr q8,[x7,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "fmla v16.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "fmla v17.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "fmla v18.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "fmla v19.4s,v8.4s,v13.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "fmla v20.4s,v8.4s,v14.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "fmla v21.4s,v8.4s,v15.4s"
        EmitIfCountGE \FilterCount\(), 2, "add x13,x7,x1"
        EmitIfCountGE \FilterCount\(), 2, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "fmla v22.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "fmla v23.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "fmla v24.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "fmla v25.4s,v8.4s,v13.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "fmla v26.4s,v8.4s,v14.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "fmla v27.4s,v8.4s,v15.4s"
        EmitIfCountGE \FilterCount\(), 3, "ldr q8,[x15,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "fmla v28.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "fmla v29.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "fmla v30.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "fmla v31.4s,v8.4s,v13.4s"
        EmitIfCountGE \FilterCount\(), 4, "add x13,x15,x1"
        EmitIfCountGE \FilterCount\(), 4, "ldr q8,[x13,#\VectorOffset\()]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "fmla v0.4s,v8.4s,v10.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "fmla v1.4s,v8.4s,v11.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "fmla v2.4s,v8.4s,v12.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "fmla v3.4s,v8.4s,v13.4s"
.endif
.endif
.endif

        .endm
//
// Generate the convolution kernels.
//

        SconvKernelFunction Nchw, 4
        SconvKernelFunction Nchwc, 4

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case of a depthwise separable convolution.

--*/

        .macro SconvKernelDepthwiseFunction

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Depthwise separable convolutions are a form of grouped convolution where
    the number of input and output channels per group are one.

Arguments:

    Input (x0) - Supplies the address of the input buffer.

    Filter (x1) - Supplies the address of the filter buffer.

    Output (x2) - Supplies the address of the output buffer.

    StrideWidth (x3) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (x4) - Supplies the length in bytes of the blocked dilation
        width.

    InputStride (x5) - Supplies the length in bytes to advance the input buffer
        to the next input row.

    KernelHeight - Supplies the height of the kernel to apply.

    KernelWidth - Supplies the width of the kernel to apply.

    InputBase - Supplies the address of the valid input buffer.

    InputWidth - Supplies the length in bytes of the blocked input width.

    DilatedInputWidth - Supplies the length in bytes to advance the input base
        buffer to the next input row including dilation.

    OutputCountLeftPad - Supplies the number of output elements that include
        one or more padding elements from the left edge.

    OutputCount - Supplies the number of output elements that do not include
        any padding elements.

    OutputCountRightPad - Supplies the number of output elements that include
        one or more padding elements from the right edge.

    Bias - Supplies the address of the bias buffer.

    Flags - Supplies additional flags controlling the convolution operation.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvDepthwiseFloatKernelNeon

        // Save callee-saved registers
        stp     x19,x20,[sp,#-80]!
        stp     x21,x22,[sp,#16]
        stp     x23,x24,[sp,#32]
        stp     x25,x26,[sp,#48]

        // Arguments:
        // x0 = Input, x1 = Filter, x2 = Output, x3 = StrideWidth
        // x4 = DilationWidth, x5 = InputStride
        // Stack: KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
        //        OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias, Flags

        mov     x19,x2                      // save Output
        mov     x2,x4                       // DilationWidth
        mov     x4,x3                       // StrideWidth
        mov     x3,x19                      // Output

//
// Process the specified number of filter rows.
//

        ProcessFilterCountN .LSconvKernelDepthwiseFrame, Depthwise, 1

//
// Restore non-volatile registers and return.
//

.LDepthwise.ExitKernel:
        ldp     x25,x26,[sp,#48]
        ldp     x23,x24,[sp,#32]
        ldp     x21,x22,[sp,#16]
        ldp     x19,x20,[sp],#80
        ret

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

MlasConvDepthwiseFloatSingleNeonFilter1:
        ProcessOutputCountN .LSconvKernelDepthwiseSingleFrame, Depthwise, 4, 1, 1
        add     x0,x0,x4                    // advance input by 1 element
        subs    x21,x21,#1                  // decrement output count remaining
        b.ne    MlasConvDepthwiseFloatSingleNeonFilter1
        ret

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel for the special
    case where the kernel dimensions are 1 (pointwise convolution).

--*/

        .macro SconvKernelPointwiseFunction

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

    Pointwise convolutions have a kernel size of one. To simplify this
    implementation, no input padding is allowed, which matches typical usage in
    models.

Arguments:

    Input (x0) - Supplies the address of the input buffer.

    Filter (x1) - Supplies the address of the filter buffer.

    Output (x2) - Supplies the address of the output buffer.

    StrideWidth (x3) - Supplies the length in bytes of the blocked stride width.

    InputChannels (x4) - Supplies the number of input channels to process.

    FilterCount (x5) - Supplies the number of rows from the filter to process.

    InputStride - Supplies the length in bytes to advance the input buffer to
        the next input channel of the same input row.

    FilterStride - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride - Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    OutputCount - Supplies the number of output elements.

    Bias - Supplies the address of the bias buffer.

    Flags - Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConvPointwiseFloatKernelNeon

        // Save callee-saved registers
        stp     x19,x20,[sp,#-80]!
        stp     x21,x22,[sp,#16]
        stp     x23,x24,[sp,#32]
        stp     x25,x26,[sp,#48]

        str     x4,[sp,#64]                 // save InputChannels locally

        // Arguments:
        // x0 = Input, x1 = Filter, x2 = Output, x3 = StrideWidth
        // x4 = InputChannels, x5 = FilterCount
        // Stack: InputStride, FilterStride, OutputStride, OutputCount, Bias, Flags

        mov     x12,x1                      // save Filter address
        ldr     x21,[sp,#104]               // OutputCount (4th stack parameter)
        ldr     x1,[sp,#88]                 // FilterStride (2nd stack parameter)
        ldr     x19,[sp,#80]                // InputStride (1st stack parameter)
        mov     x20,x5                      // save FilterCount
        mov     x4,x3                       // StrideWidth
        mov     x3,x2                       // Output

//
// Process the specified number of filter rows.
//

        cmp     x20,#3
        b.eq    .LPointwise.ProcessFilterCount3
        b.lo    .LPointwise.ProcessFilterCountLessThan3
        ProcessPointwiseFilterCountN 4
        b       .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount3:
        ProcessPointwiseFilterCountN 3
        b       .LPointwise.ExitKernel

.LPointwise.ProcessFilterCountLessThan3:
        cmp     x20,#2
        b.lo    .LPointwise.ProcessFilterCount1
        ProcessPointwiseFilterCountN 2
        b       .LPointwise.ExitKernel

.LPointwise.ProcessFilterCount1:
        ProcessPointwiseFilterCountN 1

//
// Restore non-volatile registers and return.
//

.LPointwise.ExitKernel:
        ldp     x25,x26,[sp,#48]
        ldp     x23,x24,[sp,#32]
        ldp     x21,x22,[sp,#16]
        ldp     x19,x20,[sp],#80
        ret

        .endm
        
        SconvKernelDepthwiseFunction

        SconvKernelPointwiseFunction
/*++

Macro Description:

    This macro generates code to process an output block after the inner
    convolution kernel has executed and then stores the output block to the
    output buffer.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

--*/

        .macro PostProcessBlock FilterCount, OutputCount

        .globl  MlasConvPostProcessFloatNeonFilter\FilterCount\()Output\OutputCount\()
#if !defined(__APPLE__)
        .hidden MlasConvPostProcessFloatNeonFilter\FilterCount\()Output\OutputCount\()
#endif
MlasConvPostProcessFloatNeonFilter\FilterCount\()Output\OutputCount\():

.if \FilterCount\() > 2
        add     x11,x3,x9,lsl #1            // compute output plus 2 rows
.endif

//
// Test if the existing contents of the output buffer should be accumulated
// with the output block.
//

        tst     w8,#MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT
        b.eq    .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "ldr q8,[x3]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "fadd v16.4s,v16.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "ldr q8,[x3,#16]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "fadd v17.4s,v17.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "ldr q8,[x3,#32]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "fadd v18.4s,v18.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "ldr q8,[x3,#48]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "fadd v19.4s,v19.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "ldr q8,[x3,#64]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "fadd v20.4s,v20.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "ldr q8,[x3,#80]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "fadd v21.4s,v21.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "ldr q8,[x3,x9]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "fadd v22.4s,v22.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "add x12,x3,x9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "ldr q8,[x12,#16]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "fadd v23.4s,v23.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "ldr q8,[x12,#32]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "fadd v24.4s,v24.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "ldr q8,[x12,#48]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "fadd v25.4s,v25.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "ldr q8,[x12,#64]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "fadd v26.4s,v26.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "ldr q8,[x12,#80]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "fadd v27.4s,v27.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "ldr q8,[x11]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "fadd v28.4s,v28.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "ldr q8,[x11,#16]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "fadd v29.4s,v29.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "ldr q8,[x11,#32]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "fadd v30.4s,v30.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "ldr q8,[x11,#48]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "fadd v31.4s,v31.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "add x12,x11,x9"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "ldr q8,[x12]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "fadd v0.4s,v0.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "ldr q8,[x12,#16]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "fadd v1.4s,v1.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "ldr q8,[x12,#32]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "fadd v2.4s,v2.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "ldr q8,[x12,#48]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "fadd v3.4s,v3.4s,v8.4s"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipAccumulateOutput:

//
// Test if the bias buffer should be accumulated with the output block.
//

        tst     w8,#MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION
        b.eq    .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition
.if \OutputCount\() == 1
        EmitIfCountGE \FilterCount\(), 1, "ldr q8,[x10]"
        EmitIfCountGE \FilterCount\(), 1, "fadd v16.4s,v16.4s,v8.4s"
        EmitIfCountGE \FilterCount\(), 2, "ldr q8,[x10,#16]"
        EmitIfCountGE \FilterCount\(), 2, "fadd v22.4s,v22.4s,v8.4s"
        EmitIfCountGE \FilterCount\(), 3, "ldr q8,[x10,#32]"
        EmitIfCountGE \FilterCount\(), 3, "fadd v28.4s,v28.4s,v8.4s"
        EmitIfCountGE \FilterCount\(), 4, "ldr q8,[x10,#48]"
        EmitIfCountGE \FilterCount\(), 4, "fadd v0.4s,v0.4s,v8.4s"
.else
        EmitIfCountGE \FilterCount\(), 1, "ldr q4,[x10]"
        EmitIfCountGE \FilterCount\(), 2, "ldr q5,[x10,#16]"
        EmitIfCountGE \FilterCount\(), 3, "ldr q6,[x10,#32]"
        EmitIfCountGE \FilterCount\(), 4, "ldr q7,[x10,#48]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "fadd v16.4s,v16.4s,v4.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "fadd v17.4s,v17.4s,v4.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "fadd v18.4s,v18.4s,v4.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "fadd v19.4s,v19.4s,v4.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "fadd v20.4s,v20.4s,v4.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "fadd v21.4s,v21.4s,v4.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "fadd v22.4s,v22.4s,v5.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "fadd v23.4s,v23.4s,v5.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "fadd v24.4s,v24.4s,v5.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "fadd v25.4s,v25.4s,v5.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "fadd v26.4s,v26.4s,v5.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "fadd v27.4s,v27.4s,v5.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "fadd v28.4s,v28.4s,v6.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "fadd v29.4s,v29.4s,v6.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "fadd v30.4s,v30.4s,v6.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "fadd v31.4s,v31.4s,v6.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "fadd v0.4s,v0.4s,v7.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "fadd v1.4s,v1.4s,v7.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "fadd v2.4s,v2.4s,v7.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "fadd v3.4s,v3.4s,v7.4s"
.endif

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipBiasAddition:

//
// Test for fused ReLU activation.
//

        tst     w8,#MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION
        b.eq    .LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation
        eor     v8.16b,v8.16b,v8.16b
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "fmax v16.4s,v16.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "fmax v17.4s,v17.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "fmax v18.4s,v18.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "fmax v19.4s,v19.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "fmax v20.4s,v20.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "fmax v21.4s,v21.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "fmax v22.4s,v22.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "fmax v23.4s,v23.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "fmax v24.4s,v24.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "fmax v25.4s,v25.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "fmax v26.4s,v26.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "fmax v27.4s,v27.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "fmax v28.4s,v28.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "fmax v29.4s,v29.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "fmax v30.4s,v30.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "fmax v31.4s,v31.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "fmax v0.4s,v0.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "fmax v1.4s,v1.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "fmax v2.4s,v2.4s,v8.4s"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "fmax v3.4s,v3.4s,v8.4s"

.LPostProcessBlock.\FilterCount\().\OutputCount\().SkipReluActivation:

//
// Store the output block in the output buffer.
//

        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 1, "str q16,[x3]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 2, "str q17,[x3,#16]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 3, "str q18,[x3,#32]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 4, "str q19,[x3,#48]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 5, "str q20,[x3,#64]"
        EmitIfCount2GE \FilterCount\(), 1, \OutputCount\(), 6, "str q21,[x3,#80]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 1, "str q22,[x3,x9]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "add x12,x3,x9"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 2, "str q23,[x12,#16]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 3, "str q24,[x12,#32]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 4, "str q25,[x12,#48]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 5, "str q26,[x12,#64]"
        EmitIfCount2GE \FilterCount\(), 2, \OutputCount\(), 6, "str q27,[x12,#80]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 1, "str q28,[x11]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 2, "str q29,[x11,#16]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 3, "str q30,[x11,#32]"
        EmitIfCount2GE \FilterCount\(), 3, \OutputCount\(), 4, "str q31,[x11,#48]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "add x12,x11,x9"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 1, "str q0,[x12]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 2, "str q1,[x12,#16]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 3, "str q2,[x12,#32]"
        EmitIfCount2GE \FilterCount\(), 4, \OutputCount\(), 4, "str q3,[x12,#48]"
        add     x3,x3,#\OutputCount\()*16   // advance output by N nchw4c blocks
        ret

        .endm

        .irp    FilterCount, 1, 2, 3, 4
        .irp    OutputCount, 1, 2, 3, 6
            PostProcessBlock \FilterCount\(), \OutputCount\()
        .endr
        .endr

        .end


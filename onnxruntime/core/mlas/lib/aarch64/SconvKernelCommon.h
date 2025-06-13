/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ScnvKernelNeonCommon.h

Abstract:

    This module contains common kernel macros and structures for the single
    precision convolution operation on ARM NEON.

--*/

//
// Define the convolution kernel flags.
//

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

//
// Stack frame layout for the convolution kernels.
// Local frame: 80 bytes (saved registers + filter address)
// Stack parameters start at sp+80
//

        .equ    .LSconvKernelFrame_InputStride, 80
        .equ    .LSconvKernelFrame_FilterStride, 88
        .equ    .LSconvKernelFrame_OutputStride, 96
        .equ    .LSconvKernelFrame_KernelHeight, 104
        .equ    .LSconvKernelFrame_KernelWidth, 112
        .equ    .LSconvKernelFrame_InputBase, 120
        .equ    .LSconvKernelFrame_InputWidth, 128
        .equ    .LSconvKernelFrame_DilatedInputWidth, 136
        .equ    .LSconvKernelFrame_OutputCountLeftPad, 144
        .equ    .LSconvKernelFrame_OutputCount, 152
        .equ    .LSconvKernelFrame_OutputCountRightPad, 160
        .equ    .LSconvKernelFrame_Bias, 168
        .equ    .LSconvKernelFrame_Flags, 176
        .equ    .LSconvKernelFrame_Filter, 64

        .equ    .LSconvKernelSingleFrame_InputStride, 80
        .equ    .LSconvKernelSingleFrame_FilterStride, 88
        .equ    .LSconvKernelSingleFrame_OutputStride, 96
        .equ    .LSconvKernelSingleFrame_KernelHeight, 104
        .equ    .LSconvKernelSingleFrame_KernelWidth, 112
        .equ    .LSconvKernelSingleFrame_InputBase, 120
        .equ    .LSconvKernelSingleFrame_InputWidth, 128
        .equ    .LSconvKernelSingleFrame_DilatedInputWidth, 136
        .equ    .LSconvKernelSingleFrame_OutputCountLeftPad, 144
        .equ    .LSconvKernelSingleFrame_OutputCount, 152
        .equ    .LSconvKernelSingleFrame_OutputCountRightPad, 160
        .equ    .LSconvKernelSingleFrame_Bias, 168
        .equ    .LSconvKernelSingleFrame_Flags, 176
        .equ    .LSconvKernelSingleFrame_Filter, 64

        .equ    .LSconvKernelDepthwiseFrame_KernelHeight, 80
        .equ    .LSconvKernelDepthwiseFrame_KernelWidth, 88
        .equ    .LSconvKernelDepthwiseFrame_InputBase, 96
        .equ    .LSconvKernelDepthwiseFrame_InputWidth, 104
        .equ    .LSconvKernelDepthwiseFrame_DilatedInputWidth, 112
        .equ    .LSconvKernelDepthwiseFrame_OutputCountLeftPad, 120
        .equ    .LSconvKernelDepthwiseFrame_OutputCount, 128
        .equ    .LSconvKernelDepthwiseFrame_OutputCountRightPad, 136
        .equ    .LSconvKernelDepthwiseFrame_Bias, 144
        .equ    .LSconvKernelDepthwiseFrame_Flags, 152

        .equ    .LSconvKernelDepthwiseSingleFrame_KernelHeight, 80
        .equ    .LSconvKernelDepthwiseSingleFrame_KernelWidth, 88
        .equ    .LSconvKernelDepthwiseSingleFrame_InputBase, 96
        .equ    .LSconvKernelDepthwiseSingleFrame_InputWidth, 104
        .equ    .LSconvKernelDepthwiseSingleFrame_DilatedInputWidth, 112
        .equ    .LSconvKernelDepthwiseSingleFrame_OutputCountLeftPad, 120
        .equ    .LSconvKernelDepthwiseSingleFrame_OutputCount, 128
        .equ    .LSconvKernelDepthwiseSingleFrame_OutputCountRightPad, 136
        .equ    .LSconvKernelDepthwiseSingleFrame_Bias, 144
        .equ    .LSconvKernelDepthwiseSingleFrame_Flags, 152

        .equ    .LSconvKernelPointwiseFrame_InputChannels, 64
        .equ    .LSconvKernelPointwiseFrame_InputStride, 80
        .equ    .LSconvKernelPointwiseFrame_FilterStride, 88
        .equ    .LSconvKernelPointwiseFrame_OutputStride, 96
        .equ    .LSconvKernelPointwiseFrame_OutputCount, 104
        .equ    .LSconvKernelPointwiseFrame_Bias, 112
        .equ    .LSconvKernelPointwiseFrame_Flags, 120

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks.

    OutputCount=1 generates special case code to handle padding blocks. All
    other output counts assume no padding.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    x0 - Supplies the address of the input buffer.

    x1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    x2 - Supplies the DilationWidth parameter (see function description).

    x3 - Supplies the address of the output buffer.

    x4 - Supplies the StrideWidth parameter (see function description).

    x5 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessOutputCountN KernelFrame, KernelType, BlockSize, FilterCount, OutputCount

        mov     x6,x0
.ifeqs "\KernelType\()","Depthwise"
        mov     x7,x1
.else
        ldr     x7,[sp,#\KernelFrame\()_Filter]
.endif
        ldr     x8,[sp,#\KernelFrame\()_KernelHeight]
        ldr     x9,[sp,#\KernelFrame\()_KernelWidth]
.if \OutputCount\() == 1
        ldr     x10,[sp,#\KernelFrame\()_InputBase]
        ldr     x11,[sp,#\KernelFrame\()_InputWidth]
        neg     x10,x10                     // keep negative for lea usage below
.endif
        ClearBlock \FilterCount\(), \OutputCount\()
        cbz     x8,.L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow:
        mov     x12,x9                      // reload kernel width remaining

.L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn:
.if \OutputCount\() == 1
        add     x13,x6,x10                  // compute (Input - InputBase)
        cmp     x13,x11                     // (Input - InputBase) >= InputWidth?
        b.hs    .L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding
.endif
.if \OutputCount\() > 3
        add     x14,x4,x4,lsl #1
        add     x14,x6,x14                  // compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
        add     x15,x7,x1,lsl #1            // compute filter plus 2 rows
.endif
.ifeqs "\KernelType\()","Nchwc"
.if \BlockSize\() == 4
        .irp Index, 0, 1, 2, 3
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), \Index\()*4*4, \Index\()*4
        .endr
.else
        .irp Index, 0, 1, 2, 3, 4, 5, 6, 7
            ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), \Index\()*8*4, \Index\()*4
        .endr
.endif
.else
        ComputeBlock \KernelType\(), \FilterCount\(), \OutputCount\(), 0, 0
.endif

.L\KernelType\().\FilterCount\().\OutputCount\().SkipOverPadding:
        add     x6,x6,x2                    // advance input by dilation width
.ifeqs "\KernelType\()","Nchwc"
        add     x7,x7,#\BlockSize\()*\BlockSize\()*4
                                            // advance filter by 4i4o/8i8o block
.else
        add     x7,x7,#\BlockSize\()*4      // advance filter by 4o/8o block
.endif
        subs    x12,x12,#1                  // decrement columns remaining
        b.ne    .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextColumn
        add     x6,x6,x5                    // advance input to next row
.if \OutputCount\() == 1
        ldr     x13,[sp,#\KernelFrame\()_DilatedInputWidth]
        sub     x10,x10,x13                 // advance input base to next row
.endif
        subs    x8,x8,#1                    // decrement rows remaining
        b.ne    .L\KernelType\().\FilterCount\().\OutputCount\().ProcessNextRow

//
// Handle post processing of the output block.
//

.L\KernelType\().\FilterCount\().\OutputCount\().HandlePostProcessing:
        ldr     w8,[sp,#\KernelFrame\()_Flags]
.if \FilterCount\() > 1
        ldr     x9,[sp,#\KernelFrame\()_OutputStride]
.endif
        ldr     x10,[sp,#\KernelFrame\()_Bias]
        bl      MlasConvPostProcessFloatNeonFilter\FilterCount\()Output\OutputCount\()

        .endm

/*++

Macro Description:

    This macro generates code for the inner convolution kernel.

Arguments:

    KernelType - Supplies the type of kernel to be generated.

    BlockSize - Supplies the number of elements per block.

--*/

        .macro SconvKernelFunction KernelType, BlockSize

/*++

Routine Description:

    This routine is the inner kernel to compute a convolution for the elements
    of an output row for a set of filter rows.

Arguments:

    Input (x0) - Supplies the address of the input buffer.

        The address is biased to include padding blocks for the left width
        dimension. The address is not biased to include padding rows for the
        left height dimension  these are accounted for in the outer kernel.

    Filter (x1) - Supplies the address of the filter buffer.

    Output (x2) - Supplies the address of the output buffer.

    StrideWidth (x3) - Supplies the length in bytes of the blocked stride width.

    DilationWidth (x4) - Supplies the length in bytes of the blocked dilation
        width.

    FilterCount (x5) - Supplies the number of filters to process in this
        iteration.

    InputStride - Supplies the length in bytes to advance the input buffer to
        the next input row.

    FilterStride - Supplies the length in bytes to advance the filter buffer
        to the next set of filters.

    OutputStride - Supplies the length in bytes to advance the output buffer
        to the next output address associated with the next set of filters.

    KernelHeight - Supplies the height of the kernel to apply. This height may
        be less than the original kernel height after removing any padding
        rows.

    KernelWidth - Supplies the width of the kernel to apply.

    InputBase - Supplies the address of the valid input buffer.

        This parameter is similar to the Input parameter, but does not include
        the padding blocks for the left width dimension. This parameter is used
        with the following InputWidth parameter in order to validate that the
        current input buffer address in bounds and not in the left or right
        width padding region.

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

    Flags - Supplies additional flags controlling the convolution operation,
        especially post calculation options.

Return Value:

    None.

--*/

        FUNCTION_ENTRY MlasConv\KernelType\()FloatKernelNeon

        // Save callee-saved registers
        stp     x19,x20,[sp,#-80]!
        stp     x21,x22,[sp,#16]
        stp     x23,x24,[sp,#32]
        stp     x25,x26,[sp,#48]
        str     x1,[sp,#64]                 // save filter address locally

        // Arguments:
        // x0 = Input, x1 = Filter, x2 = Output, x3 = StrideWidth
        // x4 = DilationWidth, x5 = FilterCount
        // Stack: InputStride, FilterStride, OutputStride, KernelHeight, KernelWidth,
        //        InputBase, InputWidth, DilatedInputWidth, OutputCountLeftPad,
        //        OutputCount, OutputCountRightPad, Bias, Flags

        mov     x19,x2                      // save Output
        mov     x20,x5                      // save FilterCount  
        mov     x2,x4                       // DilationWidth
        mov     x4,x3                       // StrideWidth
        mov     x3,x19                      // Output
        ldr     x5,[sp,#80]                 // InputStride (first stack parameter)
        ldr     x1,[sp,#88]                 // FilterStride (second stack parameter)

//
// Process the specified number of filter rows.
//

        cmp     x20,#3
        b.eq    .L\KernelType\().ProcessFilterCount3
        b.lo    .L\KernelType\().ProcessFilterCountLessThan3
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 4
        b       .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount3:
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 3
        b       .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCountLessThan3:
        cmp     x20,#2
        b.lo    .L\KernelType\().ProcessFilterCount1
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 2
        b       .L\KernelType\().ExitKernel

.L\KernelType\().ProcessFilterCount1:
        ProcessFilterCountN .LSconvKernelFrame, \KernelType\(), 1

//
// Restore non-volatile registers and return.
//

.L\KernelType\().ExitKernel:
        ldp     x25,x26,[sp,#48]
        ldp     x23,x24,[sp,#32]
        ldp     x21,x22,[sp,#16]
        ldp     x19,x20,[sp],#80
        ret

//
// Generate out-of-band helpers for handling output blocks involving padding.
//

        .irp FilterCount, 1, 2, 3, 4

MlasConv\KernelType\()FloatSingleNeonFilter\FilterCount\():
        ProcessOutputCountN .LSconvKernelSingleFrame, \KernelType\(), \BlockSize\(), \FilterCount\(), 1
        add     x0,x0,x4                    // advance input by 1 element
        subs    x21,x21,#1                  // decrement output count remaining
        b.ne    MlasConv\KernelType\()FloatSingleNeonFilter\FilterCount\()
        ret

        .endr

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    x0 - Supplies the address of the input buffer.

    x1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    x2 - Supplies the DilationWidth parameter (see function description).

    x3 - Supplies the address of the output buffer.

    x4 - Supplies the StrideWidth parameter (see function description).

    x5 - Supplies the InputStride parameter (see function description).

--*/

/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows.

Arguments:

    KernelFrame - Supplies the symbol name to access the convolution kernel
        stack.

    KernelType - Supplies the type of kernel to be generated.

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    x0 - Supplies the address of the input buffer.

    x1 - Supplies the FilterStride parameter (see function description) when
        KernelType!=Depthwise. Supplies the address of the filter buffer when
        KernelType=Depthwise.

    x2 - Supplies the DilationWidth parameter (see function description).

    x3 - Supplies the address of the output buffer.

    x4 - Supplies the StrideWidth parameter (see function description).

    x5 - Supplies the InputStride parameter (see function description).

--*/

        .macro ProcessFilterCountN KernelFrame, KernelType, FilterCount

//
// Process the output blocks that include left padding.
//

        ldr     x21,[sp,#\KernelFrame\()_OutputCountLeftPad]
        cbz     x21,.L\KernelType\().\FilterCount\().ProcessOutputCount
        bl      MlasConv\KernelType\()FloatSingleNeonFilter\FilterCount\()

//
// Process the output blocks that do not include any padding.
//

.L\KernelType\().\FilterCount\().ProcessOutputCount:
        ldr     x21,[sp,#\KernelFrame\()_OutputCount]
        subs    x21,x21,#6
        b.lo    .L\KernelType\().\FilterCount\().ProcessRemainingOutputCount

.L\KernelType\().\FilterCount\().ProcessNextOutputCountBy6:
        ProcessOutputCountN \KernelFrame\(), \KernelType\(), 4, \FilterCount\(), 6
        add     x22,x4,x4,lsl #1
        add     x0,x0,x22,lsl #1           // advance input by 6 elements
        subs    x21,x21,#6
        b.hs    .L\KernelType\().\FilterCount\().ProcessNextOutputCountBy6

.L\KernelType\().\FilterCount\().ProcessRemainingOutputCount:
        adds    x21,x21,#6                 // correct for over-subtract above
        b.eq    .L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining
        cmp     x21,#3
        b.lo    .L\KernelType\().\FilterCount\().ProcessRemainingOutputCountLessThan3
        ProcessOutputCountN \KernelFrame\(), \KernelType\(), 4, \FilterCount\(), 3
        add     x22,x4,x4,lsl #1
        add     x0,x0,x22                  // advance input by 3 elements
        subs    x21,x21,#3
        b.eq    .L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining

.L\KernelType\().\FilterCount\().ProcessRemainingOutputCountLessThan3:
        cmp     x21,#1
        b.eq    .L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining
        ProcessOutputCountN \KernelFrame\(), \KernelType\(), 4, \FilterCount\(), 2
        add     x0,x0,x4,lsl #1             // advance input by 2 elements
        subs    x21,x21,#2

//
// Process the output blocks that include right padding plus any remaining output
// blocks from above.
//

.L\KernelType\().\FilterCount\().ProcessOutputCountRightPadAndRemaining:
        ldr     x22,[sp,#\KernelFrame\()_OutputCountRightPad]
        add     x21,x21,x22
        cbz     x21,.L\KernelType\().ExitKernel
        bl      MlasConv\KernelType\()FloatSingleNeonFilter\FilterCount\()

        .endm
/*++

Macro Description:

    This macro generates code to compute the convolution for a specified number
    of filter rows for a pointwise convolution.

Arguments:

    FilterCount - Supplies the number of rows from the filter to process.

Implicit Arguments:

    x0 - Supplies the address of the input buffer.

    x1 - Supplies the FilterStride parameter (see function description).

    x19 - Supplies the InputStride parameter (see function description).

    x3 - Supplies the address of the output buffer.

    x4 - Supplies the StrideWidth parameter (see function description).

    x12 - Supplies the address of the filter buffer.

    x21 - Supplies the OutputCount parameter (see function description).

--*/

        .macro ProcessPointwiseFilterCountN FilterCount

        subs    x21,x21,#6
        b.lo    .LPointwise.\FilterCount\().ProcessRemainingOutputCount

.LPointwise.\FilterCount\().ProcessNextOutputCountBy6:
        ProcessPointwiseOutputCountN 4, \FilterCount\(), 6
        add     x22,x4,x4,lsl #1
        add     x0,x0,x22,lsl #1           // advance input by 6 elements
        subs    x21,x21,#6
        b.hs    .LPointwise.\FilterCount\().ProcessNextOutputCountBy6

.LPointwise.\FilterCount\().ProcessRemainingOutputCount:
        adds    x21,x21,#6                 // correct for over-subtract above
        b.eq    .LPointwise.ExitKernel
        cmp     x21,#3
        b.lo    .LPointwise.\FilterCount\().ProcessRemainingOutputCountLessThan3
        ProcessPointwiseOutputCountN 4, \FilterCount\(), 3
        add     x22,x4,x4,lsl #1
        add     x0,x0,x22                  // advance input by 3 elements
        subs    x21,x21,#3
        b.eq    .LPointwise.ExitKernel

.LPointwise.\FilterCount\().ProcessRemainingOutputCountLessThan3:
        cmp     x21,#2
        b.lo    .LPointwise.\FilterCount\().ProcessRemainingOutputCount1
        ProcessPointwiseOutputCountN 4, \FilterCount\(), 2
        b       .LPointwise.ExitKernel

.LPointwise.\FilterCount\().ProcessRemainingOutputCount1:
        ProcessPointwiseOutputCountN 4, \FilterCount\(), 1

        .endm

/*++

Macro Description:

    This macro generates code to compute the convolution for a vector of input
    blocks and a vector of filter blocks to produce a matrix of output blocks
    for a pointwise convolution.

Arguments:

    BlockSize - Supplies the number of elements per block.

    FilterCount - Supplies the number of rows from the filter to process.

    OutputCount - Supplies the number of output blocks to produce.

Implicit Arguments:

    x0 - Supplies the address of the input buffer.

    x1 - Supplies the FilterStride parameter (see function description).

    x19 - Supplies the InputStride parameter (see function description).

    x3 - Supplies the address of the output buffer.

    x4 - Supplies the StrideWidth parameter (see function description).

    x12 - Supplies the address of the filter buffer.

--*/

        .macro ProcessPointwiseOutputCountN BlockSize, FilterCount, OutputCount

        mov     x6,x0
        mov     x7,x12
        ldr     x8,[sp,#64]                 // InputChannels
        ClearBlock \FilterCount\(), \OutputCount\()

.LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock:
.if \OutputCount\() > 3
        add     x14,x4,x4,lsl #1
        add     x14,x6,x14                  // compute input plus 3 blocks
.endif
.if \FilterCount\() > 2
        add     x15,x7,x1,lsl #1            // compute filter plus 2 rows
.endif
        .irp Index, 0, 1, 2, 3
            ComputeBlock Pointwise, \FilterCount\(), \OutputCount\(), \Index\()*4*4, \Index\()*4
        .endr
        add     x6,x6,x19                   // advance input to next channel block
        add     x7,x7,#\BlockSize\()*\BlockSize\()*4
                                            // advance filter by 4i4o block
        subs    x8,x8,#1                    // decrement input blocks remaining
        b.ne    .LPointwise.\FilterCount\().\OutputCount\().ProcessNextInputBlock

//
// Handle post processing of the output block.
//

        ldr     w8,[sp,#120]                // Flags (6th stack parameter)
.if \FilterCount\() > 1
        ldr     x9,[sp,#96]                 // OutputStride (3rd stack parameter)
.endif
        ldr     x10,[sp,#112]               // Bias (5th stack parameter)
        bl      MlasConvPostProcessFloatNeonFilter\FilterCount\()Output\OutputCount\()

        .endm

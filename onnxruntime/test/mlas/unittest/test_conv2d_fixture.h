// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test_util.h"

//
// Short Execute that distinguish each test by all parameters.
//
template <typename Conv2dTester>
class Conv2dShortExecuteTest : public MlasTestFixture<Conv2dTester> {
 public:
  explicit Conv2dShortExecuteTest(size_t BatchCount,
                                  size_t GroupCount,
                                  size_t InputChannels,
                                  size_t InputHeight,
                                  size_t InputWidth,
                                  size_t FilterCount,
                                  size_t KernelHeight,
                                  size_t KernelWidth,
                                  size_t PaddingLeftHeight,
                                  size_t PaddingLeftWidth,
                                  size_t PaddingRightHeight,
                                  size_t PaddingRightWidth,
                                  size_t DilationHeight,
                                  size_t DilationWidth,
                                  size_t StrideHeight,
                                  size_t StrideWidth)
      : BatchCount_(BatchCount),
        GroupCount_(GroupCount),
        InputChannels_(InputChannels),
        InputHeight_(InputHeight),
        InputWidth_(InputWidth),
        FilterCount_(FilterCount),
        KernelHeight_(KernelHeight),
        KernelWidth_(KernelWidth),
        PaddingLeftHeight_(PaddingLeftHeight),
        PaddingLeftWidth_(PaddingLeftWidth),
        PaddingRightHeight_(PaddingRightHeight),
        PaddingRightWidth_(PaddingRightWidth),
        DilationHeight_(DilationHeight),
        DilationWidth_(DilationWidth),
        StrideHeight_(StrideHeight),
        StrideWidth_(StrideWidth) {
  }

  void TestBody() override {
    MlasTestFixture<Conv2dTester>::mlas_tester->Test(
        BatchCount_,
        GroupCount_,
        InputChannels_,
        InputHeight_,
        InputWidth_,
        FilterCount_,
        KernelHeight_,
        KernelWidth_,
        PaddingLeftHeight_,
        PaddingLeftWidth_,
        PaddingRightHeight_,
        PaddingRightWidth_,
        DilationHeight_,
        DilationWidth_,
        StrideHeight_,
        StrideWidth_);
  }

  static size_t RegisterSingleTest(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth) {
    std::stringstream ss;
    ss << "B" << BatchCount << "/"
       << "G" << GroupCount << "/"
       << "Cpg" << InputChannels << "/"
       << "Fpg" << FilterCount << "/"
       << "H" << InputHeight << "/"
       << "W" << InputWidth << "/"
       << "KH" << KernelHeight << "/"
       << "KW" << KernelWidth << "/"
       << "Pad" << PaddingLeftHeight << "," << PaddingLeftWidth << "," << PaddingRightHeight << "," << PaddingRightWidth << "/"
       << "Dilation" << DilationHeight << "," << DilationWidth << "/"
       << "Stride" << StrideHeight << "," << StrideWidth;
    auto test_name = ss.str();

    testing::RegisterTest(
        Conv2dTester::GetTestSuiteName(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> MlasTestFixture<Conv2dTester>* {
          return new Conv2dShortExecuteTest<Conv2dTester>(BatchCount,
                                                          GroupCount,
                                                          InputChannels,
                                                          InputHeight,
                                                          InputWidth,
                                                          FilterCount,
                                                          KernelHeight,
                                                          KernelWidth,
                                                          PaddingLeftHeight,
                                                          PaddingLeftWidth,
                                                          PaddingRightHeight,
                                                          PaddingRightWidth,
                                                          DilationHeight,
                                                          DilationWidth,
                                                          StrideHeight,
                                                          StrideWidth);
        });
    return 1;
  }

  static size_t RegisterShortExecuteTests() {
    size_t test_registered = 0;
    
    // Original tests
    test_registered += RegisterSingleTest(1, 1, 2, 4, 4, 4, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    test_registered += RegisterSingleTest(1, 1, 3, 4, 4, 10, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    test_registered += RegisterSingleTest(1, 16, 3, 4, 4, 10, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
    test_registered += RegisterSingleTest(1, 1, 3, 16, 16, 27, 3, 3, 3, 2, 4, 0, 2, 3, 1, 1);
    test_registered += RegisterSingleTest(1, 1, 16, 128, 128, 32, 5, 5, 1, 0, 1, 0, 3, 3, 2, 3);
    test_registered += RegisterSingleTest(1, 1, 16, 128, 128, 32, 4, 6, 2, 1, 3, 2, 4, 2, 1, 4);
    test_registered += RegisterSingleTest(3, 16, 16, 128, 128, 32, 4, 6, 2, 1, 3, 2, 4, 2, 1, 4);

    // MobileNet-specific test cases based on ONNX model analysis
    // These tests cover the exact convolution configurations found in MobileNet v2
    
    // Initial standard convolution: 3x3, stride=2, padding=1, groups=1
    // Input: [1,3,224,224] -> Output: [1,32,112,112]
    test_registered += RegisterSingleTest(1, 1, 3, 224, 224, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);
    
    // Pointwise convolutions (1x1 kernels) - various channel configurations
    test_registered += RegisterSingleTest(1, 1, 32, 112, 112, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck0_conv0
    test_registered += RegisterSingleTest(1, 1, 32, 112, 112, 16, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck0_conv2
    test_registered += RegisterSingleTest(1, 1, 16, 112, 112, 96, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck1_conv0
    test_registered += RegisterSingleTest(1, 1, 96, 56, 56, 24, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);   // linearbottleneck1_conv2
    test_registered += RegisterSingleTest(1, 1, 24, 56, 56, 144, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck2_conv0
    test_registered += RegisterSingleTest(1, 1, 144, 56, 56, 24, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck2_conv2
    test_registered += RegisterSingleTest(1, 1, 144, 28, 28, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck3_conv2
    test_registered += RegisterSingleTest(1, 1, 32, 28, 28, 192, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck4_conv0
    test_registered += RegisterSingleTest(1, 1, 192, 28, 28, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck4_conv2
    test_registered += RegisterSingleTest(1, 1, 192, 28, 28, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck6_conv2
    test_registered += RegisterSingleTest(1, 1, 64, 14, 14, 384, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck7_conv0
    test_registered += RegisterSingleTest(1, 1, 384, 14, 14, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // linearbottleneck7_conv2
    test_registered += RegisterSingleTest(1, 1, 384, 7, 7, 96, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);    // linearbottleneck10_conv2
    test_registered += RegisterSingleTest(1, 1, 96, 7, 7, 576, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);    // linearbottleneck11_conv0
    test_registered += RegisterSingleTest(1, 1, 576, 7, 7, 96, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);    // linearbottleneck11_conv2
    test_registered += RegisterSingleTest(1, 1, 576, 7, 7, 160, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);   // linearbottleneck13_conv2
    test_registered += RegisterSingleTest(1, 1, 160, 7, 7, 960, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);   // linearbottleneck14_conv0
    test_registered += RegisterSingleTest(1, 1, 960, 7, 7, 160, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);   // linearbottleneck14_conv2
    test_registered += RegisterSingleTest(1, 1, 960, 7, 7, 320, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);   // linearbottleneck16_conv2
    test_registered += RegisterSingleTest(1, 1, 320, 7, 7, 1280, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // features_conv1
    test_registered += RegisterSingleTest(1, 1, 1280, 1, 1, 1000, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1); // output_pred
    
    // Depthwise convolutions (3x3 kernels, groups = input_channels) - stride=1, padding=1
    test_registered += RegisterSingleTest(1, 32, 1, 112, 112, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);  // linearbottleneck0_conv1
    test_registered += RegisterSingleTest(1, 144, 1, 56, 56, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);   // linearbottleneck2_conv1
    test_registered += RegisterSingleTest(1, 192, 1, 28, 28, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);   // linearbottleneck4_conv1
    test_registered += RegisterSingleTest(1, 384, 1, 14, 14, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);   // linearbottleneck7_conv1
    test_registered += RegisterSingleTest(1, 576, 1, 7, 7, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);     // linearbottleneck11_conv1
    test_registered += RegisterSingleTest(1, 960, 1, 7, 7, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);     // linearbottleneck14_conv1
    
    // Depthwise convolutions (3x3 kernels, groups = input_channels) - stride=2, padding=1
    test_registered += RegisterSingleTest(1, 96, 1, 112, 112, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);  // linearbottleneck1_conv1
    test_registered += RegisterSingleTest(1, 144, 1, 56, 56, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);   // linearbottleneck3_conv1
    test_registered += RegisterSingleTest(1, 384, 1, 14, 14, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);   // linearbottleneck10_conv1
    test_registered += RegisterSingleTest(1, 576, 1, 7, 7, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);     // linearbottleneck13_conv1
    
    // Additional test cases for edge cases and variations
    // Test smaller input sizes with MobileNet patterns
    test_registered += RegisterSingleTest(1, 1, 3, 32, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);    // Small input standard conv
    test_registered += RegisterSingleTest(1, 32, 1, 16, 16, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);    // Small input depthwise conv
    test_registered += RegisterSingleTest(1, 1, 32, 16, 16, 64, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);   // Small input pointwise conv
    
    // Test batch size variations
    test_registered += RegisterSingleTest(2, 1, 3, 224, 224, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);  // Batch=2
    test_registered += RegisterSingleTest(4, 32, 1, 112, 112, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);  // Batch=4 depthwise
    
    // Test extreme channel configurations found in MobileNet
    test_registered += RegisterSingleTest(1, 1, 1280, 7, 7, 320, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);  // High channel count
    test_registered += RegisterSingleTest(1, 960, 1, 7, 7, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);     // High group count
    
    // Original loop-based tests
    for (unsigned i = 1; i < 256; i <<= 1) {
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 1, 16, i, i, 32, 1, i, 0, 0, 0, 0, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, 1, i, i, 1, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, 1, i, i, 1, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
      test_registered += RegisterSingleTest(1, 16, 1, i, i, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
      test_registered += RegisterSingleTest(1, 16, 1, i, i, 1, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2);
    }

    return test_registered;
  }

 private:
  size_t BatchCount_;
  size_t GroupCount_;
  size_t InputChannels_;
  size_t InputHeight_;
  size_t InputWidth_;
  size_t FilterCount_;
  size_t KernelHeight_;
  size_t KernelWidth_;
  size_t PaddingLeftHeight_;
  size_t PaddingLeftWidth_;
  size_t PaddingRightHeight_;
  size_t PaddingRightWidth_;
  size_t DilationHeight_;
  size_t DilationWidth_;
  size_t StrideHeight_;
  size_t StrideWidth_;
};

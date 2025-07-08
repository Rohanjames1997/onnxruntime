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
    const char* iter_env = std::getenv("MLAS_BENCHMARK_ITERATIONS");
    int benchmark_iterations = iter_env ? std::atoi(iter_env) : 100;
    int warmup_iterations = benchmark_iterations > 1 ? 10 : 0;

    if (benchmark_iterations > 1) {
      for (int i = 0; i < warmup_iterations; i++) {
        MlasTestFixture<Conv2dTester>::mlas_tester->Test(
            BatchCount_, GroupCount_, InputChannels_,
            InputHeight_, InputWidth_, FilterCount_,
            KernelHeight_, KernelWidth_,
            PaddingLeftHeight_, PaddingLeftWidth_,
            PaddingRightHeight_, PaddingRightWidth_,
            DilationHeight_, DilationWidth_,
            StrideHeight_, StrideWidth_);
      }

      auto start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < benchmark_iterations; i++) {
        MlasTestFixture<Conv2dTester>::mlas_tester->Test(
            BatchCount_, GroupCount_, InputChannels_,
            InputHeight_, InputWidth_, FilterCount_,
            KernelHeight_, KernelWidth_,
            PaddingLeftHeight_, PaddingLeftWidth_,
            PaddingRightHeight_, PaddingRightWidth_,
            DilationHeight_, DilationWidth_,
            StrideHeight_, StrideWidth_);
      }

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      double avg_time_us = duration.count() / double(benchmark_iterations);
      double ops_per_sec = 1000000.0 / avg_time_us;

      printf(
          "BENCHMARK: B%zu/G%zu/IC%zu/FC%zu/IH%zu/IW%zu/KH%zu/KW%zu/Pad%zu,%zu,%zu,%zu/Dilation%zu,%zu/Stride%zu,%zu - "
          "Avg: %.2f μs, Rate: %.2f ops/sec (%d iterations)\n",
          BatchCount_, GroupCount_, InputChannels_, FilterCount_, InputHeight_, InputWidth_,
          KernelHeight_, KernelWidth_, PaddingLeftHeight_, PaddingLeftWidth_, PaddingRightHeight_,
          PaddingRightWidth_, DilationHeight_, DilationWidth_, StrideHeight_, StrideWidth_,
          avg_time_us, ops_per_sec, benchmark_iterations);
    } else {
      MlasTestFixture<Conv2dTester>::mlas_tester->Test(
          BatchCount_, GroupCount_, InputChannels_,
          InputHeight_, InputWidth_, FilterCount_,
          KernelHeight_, KernelWidth_,
          PaddingLeftHeight_, PaddingLeftWidth_,
          PaddingRightHeight_, PaddingRightWidth_,
          DilationHeight_, DilationWidth_,
          StrideHeight_, StrideWidth_);
    }
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

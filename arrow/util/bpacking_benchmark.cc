// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <memory>
#include <stdexcept>
#include <vector>

#include <benchmark/benchmark.h>

#include "arrow/testing/util.h"
#include "arrow/util/bpacking_internal.h"
#include "arrow/util/bpacking_scalar_internal.h"
#include "arrow/util/bpacking_simd_internal.h"

#if defined(ARROW_HAVE_RUNTIME_AVX2)
#  include "arrow/util/cpu_info.h"
#endif

namespace arrow::internal {
namespace {

template <typename Int>
using UnpackFunc = void (*)(const uint8_t*, Int*, const UnpackOptions&);

/// Get the number of bytes associate with a packing.
constexpr int32_t GetNumBytes(int32_t num_values, int32_t bit_width) {
  const auto num_bits = num_values * bit_width;
  if (num_bits % 8 != 0) {
    throw std::invalid_argument("Must pack a multiple of 8 bits.");
  }
  return num_bits / 8;
}

/// Generate random bytes as packed integers.
std::vector<uint8_t> GenerateRandomPackedValues(int32_t num_values, int32_t bit_width) {
  constexpr uint32_t kSeed = 3214;
  const auto num_bytes = GetNumBytes(num_values, bit_width);

  std::vector<uint8_t> out(num_bytes);
  random_bytes(num_bytes, kSeed, out.data());

  return out;
}

template <typename Int>
void BM_Unpack(benchmark::State& state, UnpackFunc<Int> unpack, bool skip,
               std::string skip_msg) {
  if (skip) {
    state.SkipWithMessage(skip_msg);
  }

  const auto bit_width = static_cast<int32_t>(state.range(0));
  const auto num_values = static_cast<int32_t>(state.range(1));

  const auto packed = GenerateRandomPackedValues(num_values, bit_width);
  const uint8_t* packed_ptr = packed.data();

  auto unpacked = std::make_unique<Int[]>(num_values);

  const ::arrow::internal::UnpackOptions opts{
      .batch_size = num_values,
      .bit_width = bit_width,
      .bit_offset = 0,
      .max_read_bytes = -1,
  };

  for (auto _ : state) {
    unpack(packed_ptr, unpacked.get(), opts);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(num_values * state.iterations());
}

// Currently, the minimum unpack SIMD kernel size is 32 and the RLE-bit-packing encoder
// will not emit runs larger than 512 (though other implementation might), so we biased
// the benchmarks towards a rather small scale.
static const auto kNumValuesRange = benchmark::CreateRange(32, 16384, 2);
static const auto kBitWidths8 = benchmark::CreateDenseRange(0, 8, 1);
static const auto kBitWidths16 = benchmark::CreateDenseRange(0, 16, 1);
static const auto kBitWidths32 = benchmark::CreateDenseRange(0, 32, 1);
static const auto kBitWidths64 = benchmark::CreateDenseRange(0, 64, 1);

static const std::vector<std::vector<int64_t>> kBitWidthsNumValuesBool = {
    {0, 1},
    kNumValuesRange,
};
static const std::vector<std::vector<int64_t>> kBitWidthsNumValues8 = {
    kBitWidths8,
    kNumValuesRange,
};
static const std::vector<std::vector<int64_t>> kBitWidthsNumValues16 = {
    kBitWidths16,
    kNumValuesRange,
};
static const std::vector<std::vector<int64_t>> kBitWidthsNumValues32 = {
    kBitWidths32,
    kNumValuesRange,
};
static const std::vector<std::vector<int64_t>> kBitWidthsNumValues64 = {
    kBitWidths64,
    kNumValuesRange,
};

/// Nudge for MSVC template inside BENCHMARK_CAPTURE macro.
void BM_UnpackBool(benchmark::State& state, UnpackFunc<bool> unpack, bool skip = false,
                   std::string skip_msg = "") {
  return BM_Unpack<bool>(state, unpack, skip, std::move(skip_msg));
}
/// Nudge for MSVC template inside BENCHMARK_CAPTURE macro.
void BM_UnpackUint8(benchmark::State& state, UnpackFunc<uint8_t> unpack,
                    bool skip = false, std::string skip_msg = "") {
  return BM_Unpack<uint8_t>(state, unpack, skip, std::move(skip_msg));
}
/// Nudge for MSVC template inside BENCHMARK_CAPTURE macro.
void BM_UnpackUint16(benchmark::State& state, UnpackFunc<uint16_t> unpack,
                     bool skip = false, std::string skip_msg = "") {
  return BM_Unpack<uint16_t>(state, unpack, skip, std::move(skip_msg));
}
/// Nudge for MSVC template inside BENCHMARK_CAPTURE macro.
void BM_UnpackUint32(benchmark::State& state, UnpackFunc<uint32_t> unpack,
                     bool skip = false, std::string skip_msg = "") {
  return BM_Unpack<uint32_t>(state, unpack, skip, std::move(skip_msg));
}
/// Nudge for MSVC template inside BENCHMARK_CAPTURE macro.
void BM_UnpackUint64(benchmark::State& state, UnpackFunc<uint64_t> unpack,
                     bool skip = false, std::string skip_msg = "") {
  return BM_Unpack<uint64_t>(state, unpack, skip, std::move(skip_msg));
}

BENCHMARK_CAPTURE(BM_UnpackBool, NoSimd-ScalarBatch, &bpacking::unpack_scalar<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, NoSimd-ScalarBatch, &bpacking::unpack_scalar<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, NoSimd-ScalarBatch, &bpacking::unpack_scalar<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, NoSimd-ScalarBatch, &bpacking::unpack_scalar<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, NoSimd-ScalarBatch, &bpacking::unpack_scalar<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, NoSimd-ScalarExact, &bpacking::unpack_scalar_exact<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, NoSimd-ScalarExact, &bpacking::unpack_scalar_exact<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, NoSimd-ScalarExact, &bpacking::unpack_scalar_exact<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, NoSimd-ScalarExact, &bpacking::unpack_scalar_exact<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, NoSimd-ScalarExact, &bpacking::unpack_scalar_exact<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

#if defined(ARROW_HAVE_SSE4_2)
BENCHMARK_CAPTURE(BM_UnpackBool, Sse42-New, &bpacking::unpack_sse4_2<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Sse42-New, &bpacking::unpack_sse4_2<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Sse42-New, &bpacking::unpack_sse4_2<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Sse42-New, &bpacking::unpack_sse4_2<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Sse42-New, &bpacking::unpack_sse4_2<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Sse42-Old, &bpacking::unpack_sse4_2_old<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Sse42-Old, &bpacking::unpack_sse4_2_old<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Sse42-Old, &bpacking::unpack_sse4_2_old<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Sse42-Old, &bpacking::unpack_sse4_2_old<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Sse42-Old, &bpacking::unpack_sse4_2_old<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Sse42-ScalarBatch,
                  &bpacking::unpack_sse4_2_scalar_batch<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Sse42-ScalarBatch,
                  &bpacking::unpack_sse4_2_scalar_batch<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Sse42-ScalarBatch,
                  &bpacking::unpack_sse4_2_scalar_batch<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Sse42-ScalarBatch,
                  &bpacking::unpack_sse4_2_scalar_batch<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Sse42-ScalarBatch,
                  &bpacking::unpack_sse4_2_scalar_batch<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Sse42-ScalarExact, &bpacking::unpack_sse4_2_exact<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Sse42-ScalarExact, &bpacking::unpack_sse4_2_exact<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Sse42-ScalarExact, &bpacking::unpack_sse4_2_exact<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Sse42-ScalarExact, &bpacking::unpack_sse4_2_exact<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Sse42-ScalarExact, &bpacking::unpack_sse4_2_exact<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Sse42-NewNoDispatch,
                  &bpacking::unpack_sse4_2_no_dispatch<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Sse42-NewNoDispatch,
                  &bpacking::unpack_sse4_2_no_dispatch<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Sse42-NewNoDispatch,
                  &bpacking::unpack_sse4_2_no_dispatch<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Sse42-NewNoDispatch,
                  &bpacking::unpack_sse4_2_no_dispatch<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Sse42-NewNoDispatch,
                  &bpacking::unpack_sse4_2_no_dispatch<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Sse42-LittleIntPacker,
                  &bpacking::unpack_sse4_2_littleintpacker<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Sse42-LittleIntPacker,
                  &bpacking::unpack_sse4_2_littleintpacker<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Sse42-LittleIntPacker,
                  &bpacking::unpack_sse4_2_littleintpacker<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Sse42-LittleIntPacker,
                  &bpacking::unpack_sse4_2_littleintpacker<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Sse42-LittleIntPacker,
                  &bpacking::unpack_sse4_2_littleintpacker<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);
#endif

#if defined(ARROW_HAVE_RUNTIME_AVX2)
BENCHMARK_CAPTURE(BM_UnpackBool, Avx2-New, &bpacking::unpack_avx2<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx2-New, &bpacking::unpack_avx2<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx2-New, &bpacking::unpack_avx2<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx2-New, &bpacking::unpack_avx2<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx2-New, &bpacking::unpack_avx2<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Avx2-Old, &bpacking::unpack_avx2_old<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx2-Old, &bpacking::unpack_avx2_old<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx2-Old, &bpacking::unpack_avx2_old<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx2-Old, &bpacking::unpack_avx2_old<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx2-Old, &bpacking::unpack_avx2_old<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Avx2-ScalarBatch,
                  &bpacking::unpack_avx2_scalar_batch<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx2-ScalarBatch,
                  &bpacking::unpack_avx2_scalar_batch<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx2-ScalarBatch,
                  &bpacking::unpack_avx2_scalar_batch<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx2-ScalarBatch,
                  &bpacking::unpack_avx2_scalar_batch<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx2-ScalarBatch,
                  &bpacking::unpack_avx2_scalar_batch<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Avx2-ScalarExact, &bpacking::unpack_avx2_exact<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx2-ScalarExact, &bpacking::unpack_avx2_exact<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx2-ScalarExact, &bpacking::unpack_avx2_exact<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx2-ScalarExact, &bpacking::unpack_avx2_exact<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx2-ScalarExact, &bpacking::unpack_avx2_exact<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Avx2-NewNoDispatch,
                  &bpacking::unpack_avx2_no_dispatch<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx2-NewNoDispatch,
                  &bpacking::unpack_avx2_no_dispatch<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx2-NewNoDispatch,
                  &bpacking::unpack_avx2_no_dispatch<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx2-NewNoDispatch,
                  &bpacking::unpack_avx2_no_dispatch<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx2-NewNoDispatch,
                  &bpacking::unpack_avx2_no_dispatch<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX2),
                  "Avx2 not available")
    ->ArgsProduct(kBitWidthsNumValues64);
#endif

#if defined(ARROW_HAVE_RUNTIME_AVX512)
BENCHMARK_CAPTURE(BM_UnpackBool, Avx512-Old, &bpacking::unpack_avx512<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx512-Old, &bpacking::unpack_avx512<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx512-Old, &bpacking::unpack_avx512<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx512-Old, &bpacking::unpack_avx512<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx512-Old, &bpacking::unpack_avx512<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Avx512-ScalarBatch,
                  &bpacking::unpack_avx512_scalar_batch<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx512-ScalarBatch,
                  &bpacking::unpack_avx512_scalar_batch<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx512-ScalarBatch,
                  &bpacking::unpack_avx512_scalar_batch<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx512-ScalarBatch,
                  &bpacking::unpack_avx512_scalar_batch<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx512-ScalarBatch,
                  &bpacking::unpack_avx512_scalar_batch<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Avx512-ScalarExact, &bpacking::unpack_avx512_exact<bool>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Avx512-ScalarExact, &bpacking::unpack_avx512_exact<uint8_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Avx512-ScalarExact, &bpacking::unpack_avx512_exact<uint16_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Avx512-ScalarExact, &bpacking::unpack_avx512_exact<uint32_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Avx512-ScalarExact, &bpacking::unpack_avx512_exact<uint64_t>,
                  !CpuInfo::GetInstance()->IsSupported(CpuInfo::AVX512),
                  "Avx512 not available")
    ->ArgsProduct(kBitWidthsNumValues64);
#endif

#if defined(ARROW_HAVE_NEON)
BENCHMARK_CAPTURE(BM_UnpackBool, Neon-New, &bpacking::unpack_neon<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Neon-New, &bpacking::unpack_neon<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Neon-New, &bpacking::unpack_neon<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Neon-New, &bpacking::unpack_neon<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Neon-New, &bpacking::unpack_neon<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Neon-Old, &bpacking::unpack_neon_old<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Neon-Old, &bpacking::unpack_neon_old<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Neon-Old, &bpacking::unpack_neon_old<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Neon-Old, &bpacking::unpack_neon_old<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Neon-Old, &bpacking::unpack_neon_old<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Neon-ScalarBatch,
                  &bpacking::unpack_neon_scalar_batch<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Neon-ScalarBatch,
                  &bpacking::unpack_neon_scalar_batch<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Neon-ScalarBatch,
                  &bpacking::unpack_neon_scalar_batch<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Neon-ScalarBatch,
                  &bpacking::unpack_neon_scalar_batch<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Neon-ScalarBatch,
                  &bpacking::unpack_neon_scalar_batch<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Neon-ScalarExact, &bpacking::unpack_neon_exact<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Neon-ScalarExact, &bpacking::unpack_neon_exact<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Neon-ScalarExact, &bpacking::unpack_neon_exact<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Neon-ScalarExact, &bpacking::unpack_neon_exact<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Neon-ScalarExact, &bpacking::unpack_neon_exact<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

BENCHMARK_CAPTURE(BM_UnpackBool, Neon-NewNoDispatch,
                  &bpacking::unpack_neon_no_dispatch<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Neon-NewNoDispatch,
                  &bpacking::unpack_neon_no_dispatch<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Neon-NewNoDispatch,
                  &bpacking::unpack_neon_no_dispatch<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Neon-NewNoDispatch,
                  &bpacking::unpack_neon_no_dispatch<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Neon-NewNoDispatch,
                  &bpacking::unpack_neon_no_dispatch<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);
#endif

BENCHMARK_CAPTURE(BM_UnpackBool, Dynamic-Dynamic, &unpack<bool>)
    ->ArgsProduct(kBitWidthsNumValuesBool);
BENCHMARK_CAPTURE(BM_UnpackUint8, Dynamic-Dynamic, &unpack<uint8_t>)
    ->ArgsProduct(kBitWidthsNumValues8);
BENCHMARK_CAPTURE(BM_UnpackUint16, Dynamic-Dynamic, &unpack<uint16_t>)
    ->ArgsProduct(kBitWidthsNumValues16);
BENCHMARK_CAPTURE(BM_UnpackUint32, Dynamic-Dynamic, &unpack<uint32_t>)
    ->ArgsProduct(kBitWidthsNumValues32);
BENCHMARK_CAPTURE(BM_UnpackUint64, Dynamic-Dynamic, &unpack<uint64_t>)
    ->ArgsProduct(kBitWidthsNumValues64);

}  // namespace
}  // namespace arrow::internal

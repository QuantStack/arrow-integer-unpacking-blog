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

/// LittleIntPacker kernel wrappers for Arrow's unpack_width infrastructure.
///
/// These kernels wrap the hand-optimized SSE4.2 unpacking functions from LittleIntPacker:
/// Daniel Lemire and Leonid Boytsov, "Decoding billions of integers per second through
/// vectorization", Software Practice & Experience 45(1), 2015.
/// http://arxiv.org/abs/1209.2137
/// https://github.com/fast-pack/LittleIntPacker
///
/// The LittleIntPacker functions operate on uint32_t output with 128 values per kernel
/// call. Output types smaller than uint32_t are supported via a cast buffer.
///
/// Two constraints from the underlying simdhunpack functions:
/// - The output pointer must be 16-byte aligned (_mm_store_si128). Arrow's allocator
///   guarantees 64-byte alignment, and the default allocator on x86-64 returns memory
///   aligned to at least alignof(std::max_align_t) == 16, so this holds in practice.
/// - The input is over-read by up to 16 bytes past the consumed stride due to SIMD loads.
///   kBytesRead accounts for this; the consumed stride (returned by unpack()) is always
///   16 * kPackedBitSize.

#if defined(ARROW_HAVE_SSE4_2)

#  include <cstdint>
#  include <cstring>
#  include <type_traits>

#  include "LittleIntPacker/horizontalpacking32.h"

#  include "arrow/util/bpacking_dispatch_internal.h"
#  include "arrow/util/bpacking_simd_internal.h"

namespace arrow::internal::bpacking {

/// A kernel that unpacks nothing, delegating all work to the scalar implementation.
template <typename UnpackedUint, int kPackedBitSize>
struct LipNoOpKernel {
  using unpacked_type = UnpackedUint;
  static constexpr int kValuesUnpacked = 0;
  static constexpr int kBytesRead = 0;

  static const uint8_t* unpack(const uint8_t* in, UnpackedUint*) { return in; }
};

/// Compile-time dispatch to the individual simdhunpack functions.
///
/// Using direct function calls rather than the funcUnpackArr pointer table lets the
/// compiler inline each simdhunpack body into its specific instantiation of
/// LipUint32Kernel.
template <int kPackedBitSize>
void call_simdhunpack(const uint8_t* in, uint32_t* out) {
  if constexpr (kPackedBitSize == 1) simdhunpack1(in, out);
  else if constexpr (kPackedBitSize == 2) simdhunpack2(in, out);
  else if constexpr (kPackedBitSize == 3) simdhunpack3(in, out);
  else if constexpr (kPackedBitSize == 4) simdhunpack4(in, out);
  else if constexpr (kPackedBitSize == 5) simdhunpack5(in, out);
  else if constexpr (kPackedBitSize == 6) simdhunpack6(in, out);
  else if constexpr (kPackedBitSize == 7) simdhunpack7(in, out);
  else if constexpr (kPackedBitSize == 8) simdhunpack8(in, out);
  else if constexpr (kPackedBitSize == 9) simdhunpack9(in, out);
  else if constexpr (kPackedBitSize == 10) simdhunpack10(in, out);
  else if constexpr (kPackedBitSize == 11) simdhunpack11(in, out);
  else if constexpr (kPackedBitSize == 12) simdhunpack12(in, out);
  else if constexpr (kPackedBitSize == 13) simdhunpack13(in, out);
  else if constexpr (kPackedBitSize == 14) simdhunpack14(in, out);
  else if constexpr (kPackedBitSize == 15) simdhunpack15(in, out);
  else if constexpr (kPackedBitSize == 16) simdhunpack16(in, out);
  else if constexpr (kPackedBitSize == 17) simdhunpack17(in, out);
  else if constexpr (kPackedBitSize == 18) simdhunpack18(in, out);
  else if constexpr (kPackedBitSize == 19) simdhunpack19(in, out);
  else if constexpr (kPackedBitSize == 20) simdhunpack20(in, out);
  else if constexpr (kPackedBitSize == 21) simdhunpack21(in, out);
  else if constexpr (kPackedBitSize == 22) simdhunpack22(in, out);
  else if constexpr (kPackedBitSize == 23) simdhunpack23(in, out);
  else if constexpr (kPackedBitSize == 24) simdhunpack24(in, out);
  else if constexpr (kPackedBitSize == 25) simdhunpack25(in, out);
  else if constexpr (kPackedBitSize == 26) simdhunpack26(in, out);
  else if constexpr (kPackedBitSize == 27) simdhunpack27(in, out);
  else if constexpr (kPackedBitSize == 28) simdhunpack28(in, out);
  else if constexpr (kPackedBitSize == 29) simdhunpack29(in, out);
  else if constexpr (kPackedBitSize == 30) simdhunpack30(in, out);
  else if constexpr (kPackedBitSize == 31) simdhunpack31(in, out);
  else if constexpr (kPackedBitSize == 32) simdhunpack32(in, out);
}

/// Core uint32_t kernel wrapping LittleIntPacker's simdhunpack functions.
///
/// Each call unpacks 128 uint32_t values and advances the input by 16 * kPackedBitSize
/// bytes (the packed stride). The SIMD loads inside simdhunpack may read up to
/// 16 bytes past the stride, so kBytesRead is set conservatively:
///   kBytesRead = 15 * N + floor(N/2) + 16
/// which covers the maximum read address for any bit alignment.
///
/// Valid for kPackedBitSize in [1, 31].
template <int kPackedBitSize>
struct LipUint32Kernel {
  using unpacked_type = uint32_t;
  static constexpr int kValuesUnpacked = 128;

  /// Maximum bytes read (including SIMD over-read past the consumed stride).
  static constexpr int kBytesRead = 15 * kPackedBitSize + kPackedBitSize / 2 + 16;

  static const uint8_t* unpack(const uint8_t* in, uint32_t* out) {
    call_simdhunpack<kPackedBitSize>(in, out);
    return in + 16 * kPackedBitSize;
  }
};

/// Kernel that unpacks via a uint32_t buffer and casts to a narrower integer type.
///
/// Used for bool, uint8_t, and uint16_t output types, which are not natively supported
/// by LittleIntPacker.
template <typename UnpackedUint, int kPackedBitSize>
struct LipCastKernel {
  using unpacked_type = UnpackedUint;
  static constexpr int kValuesUnpacked = LipUint32Kernel<kPackedBitSize>::kValuesUnpacked;
  static constexpr int kBytesRead = LipUint32Kernel<kPackedBitSize>::kBytesRead;

  static const uint8_t* unpack(const uint8_t* in, UnpackedUint* out) {
    uint32_t buffer[kValuesUnpacked];
    in = LipUint32Kernel<kPackedBitSize>::unpack(in, buffer);
    for (int k = 0; k < kValuesUnpacked; ++k) {
      out[k] = static_cast<UnpackedUint>(buffer[k]);
    }
    return in;
  }
};

/// Select the right kernel given output type and packed bit size.
///
/// Dispatch table:
/// - bool, kPackedBitSize == 1        → LipCastKernel<bool, 1>
/// - uint8_t, kPackedBitSize in 1-7   → LipCastKernel<uint8_t, N>
/// - uint16_t, kPackedBitSize in 1-15 → LipCastKernel<uint16_t, N>
/// - uint32_t, kPackedBitSize in 1-31 → LipUint32Kernel<N>
/// - uint64_t, kPackedBitSize in 1-32 → LipCastKernel<uint64_t, N> (via uint32_t buffer)
/// - everything else                  → LipNoOpKernel (scalar fallback)
template <typename UnpackedUint, int kPackedBitSize>
constexpr auto LipKernelDispatchImpl() {
  if constexpr (std::is_same_v<UnpackedUint, bool>) {
    return LipCastKernel<bool, 1>{};
  } else if constexpr (sizeof(UnpackedUint) < sizeof(uint64_t)) {
    // uint8_t, uint16_t, uint32_t
    if constexpr (std::is_same_v<UnpackedUint, uint32_t>) {
      return LipUint32Kernel<kPackedBitSize>{};
    } else {
      return LipCastKernel<UnpackedUint, kPackedBitSize>{};
    }
  } else {
    // uint64_t: packed values fit in uint32_t for bit sizes 1-32, so cast via uint32_t
    // kernel
    if constexpr (kPackedBitSize >= 1 && kPackedBitSize <= 32) {
      return LipCastKernel<uint64_t, kPackedBitSize>{};
    } else {
      return LipNoOpKernel<uint64_t, kPackedBitSize>{};
    }
  }
}

/// Public kernel template for use as the ``Unpacker`` argument to ``unpack_jump``.
template <typename UnpackedUint, int kPackedBitSize>
using LipKernel = decltype(LipKernelDispatchImpl<UnpackedUint, kPackedBitSize>());

template <typename Uint>
void unpack_sse4_2_littleintpacker(const uint8_t* in, Uint* out,
                                   const UnpackOptions& opts) {
  return unpack_jump<LipKernel>(in, out, opts);
}

template void unpack_sse4_2_littleintpacker<bool>(const uint8_t*, bool*,
                                                  const UnpackOptions&);
template void unpack_sse4_2_littleintpacker<uint8_t>(const uint8_t*, uint8_t*,
                                                     const UnpackOptions&);
template void unpack_sse4_2_littleintpacker<uint16_t>(const uint8_t*, uint16_t*,
                                                      const UnpackOptions&);
template void unpack_sse4_2_littleintpacker<uint32_t>(const uint8_t*, uint32_t*,
                                                      const UnpackOptions&);
template void unpack_sse4_2_littleintpacker<uint64_t>(const uint8_t*, uint64_t*,
                                                      const UnpackOptions&);

}  // namespace arrow::internal::bpacking

#endif  // ARROW_HAVE_SSE4_2

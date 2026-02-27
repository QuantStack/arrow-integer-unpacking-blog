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

#if defined(ARROW_HAVE_NEON)
#  define UNPACK_PLATFORM_OLD unpack_neon_old
#elif defined(ARROW_HAVE_SSE4_2)
#  define UNPACK_PLATFORM_OLD unpack_sse4_2_old
#endif

#if defined(UNPACK_PLATFORM_OLD)

#  include "arrow/util/bpacking_dispatch_internal.h"
#  include "arrow/util/bpacking_simd128_generated_internal.h"
#  include "arrow/util/bpacking_simd_internal.h"

namespace arrow::internal::bpacking {

template <typename Uint>
void UNPACK_PLATFORM_OLD(const uint8_t* in, Uint* out, const UnpackOptions& opts) {
  return unpack_jump<Simd128UnpackerForWidth>(in, out, opts);
}

template void UNPACK_PLATFORM_OLD<bool>(const uint8_t*, bool*, const UnpackOptions&);
template void UNPACK_PLATFORM_OLD<uint8_t>(const uint8_t*, uint8_t*,
                                           const UnpackOptions&);
template void UNPACK_PLATFORM_OLD<uint16_t>(const uint8_t*, uint16_t*,
                                            const UnpackOptions&);
template void UNPACK_PLATFORM_OLD<uint32_t>(const uint8_t*, uint32_t*,
                                            const UnpackOptions&);
template void UNPACK_PLATFORM_OLD<uint64_t>(const uint8_t*, uint64_t*,
                                            const UnpackOptions&);

}  // namespace arrow::internal::bpacking

#  undef UNPACK_PLATFORM_OLD
#endif  // UNPACK_PLATFORM_OLD

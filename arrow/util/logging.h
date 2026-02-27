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

#pragma once

#include <cassert>

namespace arrow::internal {

/// Sink that silently discards anything streamed into it via <<.
/// Used so that ARROW_CHECK(cond) << "message" compiles even though
/// assert() returns void.
struct NullLog {
  template <typename T>
  [[maybe_unused]] const NullLog& operator<<([[maybe_unused]] const T&) const {
    return *this;
  }
};

[[maybe_unused]] inline constexpr NullLog kNullLog{};

}  // namespace arrow::internal

#define ARROW_CHECK(condition) (assert(condition), ::arrow::internal::kNullLog)
#define ARROW_CHECK_OK(s) (assert((s).ok()), ::arrow::internal::kNullLog)
#define ARROW_CHECK_EQ(a, b) (assert((a) == (b)), ::arrow::internal::kNullLog)
#define ARROW_CHECK_NE(a, b) (assert((a) != (b)), ::arrow::internal::kNullLog)
#define ARROW_CHECK_LE(a, b) (assert((a) <= (b)), ::arrow::internal::kNullLog)
#define ARROW_CHECK_LT(a, b) (assert((a) < (b)), ::arrow::internal::kNullLog)
#define ARROW_CHECK_GE(a, b) (assert((a) >= (b)), ::arrow::internal::kNullLog)
#define ARROW_CHECK_GT(a, b) (assert((a) > (b)), ::arrow::internal::kNullLog)

#define ARROW_LOG(level) ::arrow::internal::kNullLog

#define ARROW_DCHECK ARROW_CHECK
#define ARROW_DCHECK_OK ARROW_CHECK_OK
#define ARROW_DCHECK_EQ ARROW_CHECK_EQ
#define ARROW_DCHECK_NE ARROW_CHECK_NE
#define ARROW_DCHECK_LE ARROW_CHECK_LE
#define ARROW_DCHECK_LT ARROW_CHECK_LT
#define ARROW_DCHECK_GE ARROW_CHECK_GE
#define ARROW_DCHECK_GT ARROW_CHECK_GT

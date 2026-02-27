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

#include "arrow/testing/util.h"

#include <cstdint>
#include <random>
#include <vector>

namespace arrow {

void random_bytes(int64_t n, uint32_t seed, uint8_t* out) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint8_t>::max());
  for (int64_t i = 0; i < n; ++i) {
    out[i] = static_cast<uint8_t>(dist(rng));
  }
}

void random_is_valid(int64_t n, double pct_null, std::vector<bool>* is_valid,
                     int random_seed) {
  std::mt19937 rng(static_cast<uint32_t>(random_seed));
  std::bernoulli_distribution dist(1.0 - pct_null);
  is_valid->resize(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < is_valid->size(); ++i) {
    (*is_valid)[i] = dist(rng);
  }
}

}  // namespace arrow

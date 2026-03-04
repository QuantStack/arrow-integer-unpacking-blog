#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <xcpp/xdisplay.hpp>
#include <xwidgets/xall.hpp>

#include "arrow/util/bit_util.h"
#include "arrow/util/bpacking_dispatch_internal.h"
#include "arrow/util/bpacking_internal.h"
#include "arrow/util/bpacking_simd_kernel_internal.h"

#include "blog/components.hpp"
#include "blog/ui.hpp"
#include "blog/utils.hpp"

// Since we are not linking with libarrow, we need to add all the C++ code
// that is indirectly depended upon by header inclusion
#include "arrow/util/bpacking_scalar.cc"
#include "arrow/util/bpacking_simd_default.cc"

#pragma once

#include <charconv>
#include <concepts>
#include <cstdint>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <vector>

template <std::unsigned_integral UintType = std::size_t, std::ranges::input_range Range>
auto MaxWidthBits(Range&& in) -> std::size_t {
  auto bit_width = [](auto x) { return std::bit_width(x); };
  const auto value = std::ranges::max(in, {}, bit_width);
  return static_cast<UintType>(bit_width(value));
}

template <typename Uint = std::size_t>
auto ParseCsvInt(std::string_view s) -> std::vector<Uint> {
  std::vector<Uint> out = {};
  std::size_t pos = 0;

  while (true) {
    pos = s.find_first_not_of(' ', pos);
    if (pos == std::string_view::npos) {
      throw std::invalid_argument("unexpected end");
    }

    Uint v = 0;
    const char* b = s.data() + pos;
    const char* e = s.data() + s.size();
    auto [p, ec] = std::from_chars(b, e, v);
    if (ec != std::errc()) {
      throw std::invalid_argument("invalid integer");
    }
    out.push_back(v);
    pos = static_cast<size_t>(p - s.data());

    pos = s.find_first_not_of(' ', pos);
    if (pos == std::string_view::npos) {
      break;
    }
    if (s[pos] != ',') {
      throw std::invalid_argument("expected comma");
    }
    ++pos;
  }

  return out;
}

enum struct Uint {
  u8 = 8,
  u16 = 16,
  u32 = 32,
  u64 = 64,
};

template <Uint kUint>
using UintType = std::conditional_t<
    kUint == Uint::u8, std::uint8_t,
    std::conditional_t<
        kUint == Uint::u16, std::uint16_t,
        std::conditional_t<kUint == Uint::u32, std::uint32_t, std::uint64_t>>>;

constexpr auto ParseUint(std::string_view s) -> Uint {
  constexpr std::string_view prefix = "uint";
  constexpr std::string_view suffix = "_t";
  if (!s.starts_with(prefix) || !s.ends_with(suffix)) {
    throw std::invalid_argument("expected uint like name (e.g. uint32_t)");
  }

  s.remove_prefix(prefix.size());
  s.remove_suffix(suffix.size());

  std::uint64_t val = 0;
  auto [_, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
  if (ec != std::errc()) {
    throw std::invalid_argument("invalid integer");
  }

  return static_cast<Uint>(val);
}

constexpr auto UintMax(Uint u) -> std::uint64_t {
  constexpr auto kOnes = ~std::uint64_t{0};
  return kOnes >> (8 * sizeof(std::uint64_t) - static_cast<std::uint8_t>(u));
}

inline auto RandomValues(std::size_t n, std::size_t packed_bit_size,
                         std::mt19937::result_type seed = 33)
    -> std::vector<std::uint64_t> {
  std::mt19937 gen(seed);

  // Calculate max value that fits in packed_bit_size bits
  const auto max_value = (std::uint64_t{1} << packed_bit_size) - std::uint64_t{1};
  std::uniform_int_distribution<std::uint64_t> dist(0, max_value);

  auto result = std::vector<std::uint64_t>();
  result.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(dist(gen));
  }
  return result;
}

struct MakeAlphaValuesParams {
  std::size_t packed_bit_size;
  std::size_t n_values;
  std::optional<std::size_t> total_bits = std::nullopt;
  std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  char extra_bit = '*';
};

inline auto MakeAlphaValues(const MakeAlphaValuesParams& params) -> std::vector<char> {
  const auto asked_letter_bits = params.packed_bit_size * params.n_values;
  const auto total_bits = params.total_bits.value_or(asked_letter_bits);
  auto out = std::vector<char>(total_bits, params.extra_bit);

  const auto letter_bits = std::min(total_bits, asked_letter_bits);
  for (std::size_t i = 0; i < letter_bits; ++i) {
    const auto val_idx = i / params.packed_bit_size;
    out[i] = params.alphabet[val_idx % params.alphabet.size()];
  }

  return out;
}

/// Jump table generation for compile-time to runtime dispatch
/// Generates jump tables for functions templated on <int N, Uint U>
/// Function signature: void Func<int, Uint>::call()
/// Note: N starts from 1, not 0 (0 is invalid)
template <template <int, Uint> typename Func, Uint U, std::size_t... Is>
constexpr auto MakeJumpTableImpl(std::index_sequence<Is...>) {
  using FuncPtr = void (*)();
  return std::array<FuncPtr, sizeof...(Is)>{
      +[]() { Func<static_cast<int>(Is) + 1, U>::call(); }...};
}

template <template <int, Uint> typename Func, Uint U, std::size_t MaxN>
constexpr auto MakeJumpTable() {
  return MakeJumpTableImpl<Func, U>(std::make_index_sequence<MaxN>{});
}

/// Runtime dispatch wrapper for jump tables
/// Creates jump tables internally and dispatches to the appropriate function
/// Note: packed_bit_size must be in range [1, uint], 0 is invalid
template <template <int, Uint> typename Func>
void Dispatch(int packed_bit_size, Uint uint) {
  constexpr auto table_u8 = MakeJumpTable<Func, Uint::u8, static_cast<int>(Uint::u8)>();
  constexpr auto table_u16 =
      MakeJumpTable<Func, Uint::u16, static_cast<int>(Uint::u16)>();
  constexpr auto table_u32 =
      MakeJumpTable<Func, Uint::u32, static_cast<int>(Uint::u32)>();
  constexpr auto table_u64 =
      MakeJumpTable<Func, Uint::u64, static_cast<int>(Uint::u64)>();

  if (packed_bit_size < 1 || packed_bit_size > static_cast<int>(uint)) {
    throw std::invalid_argument("invalid n for given Uint type");
  }
  switch (uint) {
    case Uint::u8:
      table_u8[packed_bit_size - 1]();
      break;
    case Uint::u16:
      table_u16[packed_bit_size - 1]();
      break;
    case Uint::u32:
      table_u32[packed_bit_size - 1]();
      break;
    case Uint::u64:
      table_u64[packed_bit_size - 1]();
      break;
  }
}

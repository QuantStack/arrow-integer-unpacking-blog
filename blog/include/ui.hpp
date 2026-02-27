#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <ranges>

#include "arrow/util/endian.h"

inline constexpr std::size_t kBitsPerBytes = 8;

/************************
 *  Color Manipulation  *
 ************************/

struct HSL {
  /// Between [0, 360[
  double hue = 0.0;
  /// Between [0, 1]
  double saturation = 0.0;
  /// Between [0, 1]
  double lightness = 0.0;
};

struct RGB {
  std::uint8_t red = 0;
  std::uint8_t green = 0;
  std::uint8_t blue = 0;

  constexpr static auto FromHex(std::size_t hex) -> RGB {
    return {
        .red = static_cast<std::uint8_t>((hex >> 8 * 2) & 0xFF),
        .green = static_cast<std::uint8_t>((hex >> 8 * 1) & 0xFF),
        .blue = static_cast<std::uint8_t>((hex >> 8 * 0) & 0xFF),
    };
  }

  static auto FromHSL(const HSL& hsl) -> RGB {
    const double h = hsl.hue;
    const double s = hsl.saturation;
    const double l = hsl.lightness;

    const double c = (1.0 - std::abs(2.0 * l - 1.0)) * s;
    const double h_prime = h / 60.0;
    const double x = c * (1.0 - std::abs(std::fmod(h_prime, 2.0) - 1.0));
    const double m = l - c / 2.0;

    double r1 = 0.0, g1 = 0.0, b1 = 0.0;
    if (h_prime >= 0.0 && h_prime < 1.0) {
      r1 = c;
      g1 = x;
      b1 = 0.0;
    } else if (h_prime >= 1.0 && h_prime < 2.0) {
      r1 = x;
      g1 = c;
      b1 = 0.0;
    } else if (h_prime >= 2.0 && h_prime < 3.0) {
      r1 = 0.0;
      g1 = c;
      b1 = x;
    } else if (h_prime >= 3.0 && h_prime < 4.0) {
      r1 = 0.0;
      g1 = x;
      b1 = c;
    } else if (h_prime >= 4.0 && h_prime < 5.0) {
      r1 = x;
      g1 = 0.0;
      b1 = c;
    } else if (h_prime >= 5.0 && h_prime < 6.0) {
      r1 = c;
      g1 = 0.0;
      b1 = x;
    }

    return {
        .red = static_cast<std::uint8_t>(std::round((r1 + m) * 255.0)),
        .green = static_cast<std::uint8_t>(std::round((g1 + m) * 255.0)),
        .blue = static_cast<std::uint8_t>(std::round((b1 + m) * 255.0)),
    };
  }

  auto ToHSL() const -> HSL {
    const double r = red / 255.0;
    const double g = green / 255.0;
    const double b = blue / 255.0;

    const double max_val = std::max({r, g, b});
    const double min_val = std::min({r, g, b});
    const double delta = max_val - min_val;

    HSL hsl;
    hsl.lightness = (max_val + min_val) / 2.0;

    if (delta < 1e-10) {
      // Achromatic (gray)
      hsl.hue = 0.0;
      hsl.saturation = 0.0;
    } else {
      // Calculate saturation
      hsl.saturation = delta / (1.0 - std::abs(2.0 * hsl.lightness - 1.0));

      // Calculate hue
      if (max_val == r) {
        hsl.hue = 60.0 * std::fmod((g - b) / delta, 6.0);
      } else if (max_val == g) {
        hsl.hue = 60.0 * ((b - r) / delta + 2.0);
      } else {
        hsl.hue = 60.0 * ((r - g) / delta + 4.0);
      }

      // Ensure hue is in [0, 360[
      if (hsl.hue < 0.0) {
        hsl.hue += 360.0;
      }
    }

    return hsl;
  }
};

inline constexpr auto WHITE = RGB::FromHex(0xFFFFFF);
inline constexpr auto BLACK = RGB::FromHex(0x000000);
inline constexpr auto LIGHT_GREY = RGB::FromHex(0xD9D9D9);
inline constexpr auto DARK_GREY = RGB::FromHex(0x525252);
inline constexpr auto DARK_ORANGE = RGB::FromHex(0XB34B00);
inline constexpr auto LIGHT_ORANGE = RGB::FromHex(0xFFC69D);
inline constexpr auto DARK_BLUE = RGB::FromHex(0x177C9B);
inline constexpr auto LIGHT_BLUE = RGB::FromHex(0xB8E6F4);

struct TextStyle {
  std::optional<RGB> fg = std::nullopt;
  std::optional<RGB> bg = std::nullopt;
};

inline auto MakePaletteStyle(const HSL& base_color, std::size_t n)
    -> std::vector<TextStyle> {
  auto out = std::vector<TextStyle>();
  out.reserve(n);
  const auto n_pairs = (n + 1) / 2;
  for (std::size_t i = 0; i < n; ++i) {
    HSL color = base_color;
    const auto pair_idx = i / 2;
    const auto is_odd = i % 2;
    color.hue =
        std::fmod(base_color.hue + (360.0 * pair_idx / n_pairs) + is_odd * 180.0, 360.0);
    const auto bg = RGB::FromHSL(color);
    const auto fg = color.lightness < 0.5 ? WHITE : BLACK;
    out.push_back({.fg = fg, .bg = bg});
  }
  return out;
}

inline auto MakePaletteStyle(const RGB& base_color, std::size_t n) {
  return MakePaletteStyle(base_color.ToHSL(), n);
}

inline void PrintColored(auto text, const TextStyle& style = {},
                         std::ostream& out = std::cout) {
  if (style.fg) {
    out << "\033[38;2;" << static_cast<int>(style.fg->red) << ";"
        << static_cast<int>(style.fg->green) << ";" << static_cast<int>(style.fg->blue)
        << "m";
  }
  if (style.bg) {
    out << "\033[48;2;" << static_cast<int>(style.bg->red) << ";"
        << static_cast<int>(style.bg->green) << ";" << static_cast<int>(style.bg->blue)
        << "m";
  }
  out << text;
  if (style.fg || style.bg) {
    out << "\033[0m";
  }
}

inline auto NoTextStyle(std::size_t) -> TextStyle { return {}; }

struct PrintBytesParams {
  std::optional<std::size_t> lane_bit_size = std::nullopt;
  std::string lane_end = "";
  std::function<TextStyle(std::size_t)> bit_style = &NoTextStyle;
  std::optional<std::size_t> max_bytes = std::nullopt;
};

template <typename Range>
  requires std::ranges::input_range<Range>
void PrintBytes(Range&& values, const PrintBytesParams& params,
                std::ostream& out = std::cout) {
  using value_type = std::decay_t<std::ranges::range_value_t<Range>>;
  constexpr auto kBitsPerByte = std::size_t{8};
  constexpr auto kBytePerValue = sizeof(value_type);
  constexpr auto kFirstBitMask = std::uint8_t{0b1};

  const auto lane_bit_size =
      params.lane_bit_size.value_or(std::numeric_limits<std::size_t>::max());

  auto out_bit_idx = std::size_t{0};
  for (const value_type val_any : values) {
    // Respect limit on number of bytes
    if (params.max_bytes && out_bit_idx / kBitsPerBytes >= *params.max_bytes) {
      break;
    }

    const auto val_le = arrow::bit_util::ToLittleEndian<value_type>(val_any);
    for (std::size_t i = 0; i < kBytePerValue; ++i) {
      // Print byte separator at beginning of lane
      if (out_bit_idx % lane_bit_size == 0) {
        out << "│";
      }

      const std::uint8_t byte = reinterpret_cast<const std::uint8_t*>(&val_le)[i];
      for (std::size_t j = 0; j < kBitsPerByte; ++j) {
        const std::uint32_t bit = (byte >> j) & kFirstBitMask;
        const auto style = params.bit_style(out_bit_idx);
        PrintColored(bit, style, out);
        ++out_bit_idx;
      }

      // Print byte separator at end of byte
      out << "│";
      // Print lane end at end of lane
      if (out_bit_idx % lane_bit_size == 0) {
        out << params.lane_end;
      }
    }
  }
}

template <typename Range>
  requires std::ranges::input_range<Range>
void PrintValuesAsBits(Range&& values, const PrintBytesParams& params,
                       std::ostream& out = std::cout) {
  using value_type = std::decay_t<std::ranges::range_value_t<Range>>;

  const auto lane_bit_size =
      params.lane_bit_size.value_or(std::numeric_limits<std::size_t>::max());

  auto out_bit_idx = std::size_t{0};
  for (const value_type val : values) {
    // Respect limit on number of bytes
    if (params.max_bytes && out_bit_idx / kBitsPerBytes >= *params.max_bytes) {
      break;
    }

    // Print byte separator at beginning of lane
    if (out_bit_idx % lane_bit_size == 0) {
      out << "│";
    }

    const auto style = params.bit_style(out_bit_idx);
    PrintColored(val, style, out);
    ++out_bit_idx;

    // Print byte separator at end of byte
    if (out_bit_idx % kBitsPerBytes == 0) {
      out << "│";
    }
    // Print lane end at end of lane
    if (out_bit_idx % lane_bit_size == 0) {
      out << params.lane_end;
    }
  }
}

template <typename Uint>
  requires std::unsigned_integral<Uint>
void PrintBytes(Uint value, const PrintBytesParams& params,
                std::ostream& out = std::cout) {
  auto* const bytes = reinterpret_cast<const std::uint8_t*>(&value);
  return PrintBytes(std::span(bytes, sizeof(Uint)), params, out);
}

struct PrintPackedColorParams {
  std::size_t packed_bit_size;
  std::optional<std::size_t> n_valid_bits = std::nullopt;
  std::function<bool(std::size_t)> bit_highlight = [](auto) { return false; };
  std::vector<TextStyle> styles = {{.fg = BLACK, .bg = LIGHT_ORANGE},
                                   {.fg = BLACK, .bg = LIGHT_BLUE}};
  std::vector<TextStyle> highlight_styles = {{.fg = WHITE, .bg = DARK_ORANGE},
                                             {.fg = WHITE, .bg = DARK_BLUE}};
};

struct PrintPackedSelectColorParams {
  std::size_t packed_bit_size;
  std::size_t selected_value_idx;
  std::optional<std::size_t> n_valid_bits = std::nullopt;
};

template <std::ranges::range Range>
inline auto MakeColorFn(Range&& range) {
  auto styles =
      std::vector<TextStyle>(std::ranges::begin(range), std::ranges::end(range));
  return
      [styles = std::move(styles)](std::size_t k) -> TextStyle { return styles.at(k); };
}

inline auto MakePackedColorFn(const PrintPackedColorParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    // Avoid coloring past given the valid number of bits
    if (!params.n_valid_bits || bit_idx < params.n_valid_bits) {
      const auto value_idx = bit_idx / params.packed_bit_size;
      const auto style_idx = value_idx % params.styles.size();

      if (params.bit_highlight(bit_idx)) {
        return params.highlight_styles[style_idx];
      }
      return params.styles[style_idx];
    }
    return {};
  };
}

inline auto MakePackedColorFn(const PrintPackedSelectColorParams& params) {
  return MakePackedColorFn({
      .packed_bit_size = params.packed_bit_size,
      .n_valid_bits = params.n_valid_bits,
      .bit_highlight = [=](std::size_t bit_idx) -> bool {
        return bit_idx / params.packed_bit_size == params.selected_value_idx;
      },
  });
}

struct PrintUnpackedColorParams {
  std::size_t packed_bit_size;
  std::size_t unpacked_bit_size;
  std::function<bool(std::size_t)> bit_highlight = [](auto) { return false; };
  std::optional<std::size_t> n_colored_values = std::nullopt;
  std::vector<TextStyle> styles = {{.fg = BLACK, .bg = LIGHT_ORANGE},
                                   {.fg = BLACK, .bg = LIGHT_BLUE}};
  std::vector<TextStyle> highlight_styles = {{.fg = WHITE, .bg = DARK_ORANGE},
                                             {.fg = WHITE, .bg = DARK_BLUE}};
};

struct PrintUnpackedSelectColorParams {
  std::size_t packed_bit_size;
  std::size_t unpacked_bit_size;
  std::size_t selected_value_idx;
  std::optional<std::size_t> n_colored_values = std::nullopt;
};

inline auto MakeUnpackedColorFn(const PrintUnpackedColorParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    const auto value_idx = bit_idx / params.unpacked_bit_size;
    const auto bit_idx_in_val = bit_idx % params.unpacked_bit_size;

    // Respect non colored values
    if (params.n_colored_values && value_idx >= *params.n_colored_values) {
      return {};
    }
    // Color the packed bits
    if (bit_idx_in_val < params.packed_bit_size) {
      const auto style_idx = value_idx % params.styles.size();

      if (params.bit_highlight(bit_idx)) {
        return params.highlight_styles[style_idx];
      }
      return params.styles[style_idx];
    }
    // Leave padding uncolored
    return {};
  };
}

inline auto MakeUnpackedColorFn(const PrintUnpackedSelectColorParams& params) {
  return MakeUnpackedColorFn({
      .packed_bit_size = params.packed_bit_size,
      .unpacked_bit_size = params.unpacked_bit_size,
      .bit_highlight = [=](std::size_t bit_idx) -> bool {
        return bit_idx / params.unpacked_bit_size == params.selected_value_idx;
      },
      .n_colored_values = params.n_colored_values,
  });
}

/**********************************
 *  Exact unpacking UI utilities  *
 **********************************/

struct UnpackExactSelectedParams {
  std::size_t value_idx;
  std::size_t n_values;
  std::size_t packed_bit_size;
  std::size_t unpacked_bit_size;
  std::size_t max_spread_bytes;
  std::vector<TextStyle> styles = {{.fg = BLACK, .bg = LIGHT_ORANGE},
                                   {.fg = BLACK, .bg = LIGHT_BLUE}};
  std::vector<TextStyle> highlight_styles = {{.fg = WHITE, .bg = DARK_ORANGE},
                                             {.fg = WHITE, .bg = DARK_BLUE}};
  TextStyle spread_style = {.fg = BLACK, .bg = LIGHT_GREY};
  TextStyle mask_style = {.fg = WHITE, .bg = DARK_GREY};

  constexpr auto NBits() const -> std::size_t {  //
    return n_values * packed_bit_size;
  }
  constexpr auto NBytes() const -> std::size_t {
    const auto n_bits = NBits();
    return (n_bits / kBitsPerBytes) + (n_bits % kBitsPerBytes != 0);
  }
  constexpr auto MaxSpreadBits() const -> std::size_t {
    return max_spread_bytes * kBitsPerBytes;
  }
  constexpr auto ValueBitStart() const -> std::size_t {
    return packed_bit_size * value_idx;
  }
  constexpr auto ValueByteStart() const -> std::size_t {
    return ValueBitStart() / kBitsPerBytes;
  }
  constexpr auto ValueSpreadByteStart() const -> std::size_t {  //
    return ValueByteStart();
  }
  constexpr auto ValueSpreadBitStart() const -> std::size_t {
    // Not the same as ValueBitStart
    return ValueSpreadByteStart() * kBitsPerBytes;
  }
  constexpr auto ValueSpreadByteEnd() const -> std::size_t {
    return ValueSpreadByteStart() + max_spread_bytes;
  }
  constexpr auto ValueSpreadBitEnd() const -> std::size_t {
    return ValueSpreadByteEnd() * kBitsPerBytes;
  }
  constexpr auto SpreadShiftBits() const -> std::size_t {
    return ValueBitStart() - ValueSpreadBitStart();
  }
};

inline auto MakePackedSelectColorFn(const UnpackExactSelectedParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    const auto value_idx = bit_idx / params.packed_bit_size;
    if (value_idx == params.value_idx) {
      const auto style_idx = value_idx % params.highlight_styles.size();
      return params.highlight_styles[style_idx];
    }
    const auto byte_idx = bit_idx / 8;
    if (byte_idx >= params.ValueByteStart() && byte_idx < params.ValueSpreadByteEnd()) {
      return params.spread_style;
    }
    return {};
  };
}

template <typename Range>
  requires std::ranges::input_range<Range>
void PrintPackedSelect(Range&& values, const UnpackExactSelectedParams& params,
                       std::ostream& out = std::cout) {
  return PrintBytes(std::forward<Range>(values),
                    {.bit_style = MakePackedSelectColorFn(params)});
}

inline auto MakeBufferSelectColorFn(const UnpackExactSelectedParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    // We look up the value as it would be in the full buffer
    const auto bit_off_idx = bit_idx + params.ValueSpreadBitStart();
    const auto value_off_idx = bit_off_idx / params.packed_bit_size;
    if (value_off_idx == params.value_idx) {
      const auto style_idx = value_off_idx % params.highlight_styles.size();
      return params.highlight_styles[style_idx];
    }
    // Values were out of input buffer so we show them as padded zeros
    if (bit_off_idx >= kBitsPerBytes * params.NBytes()) {
      return {};
    }
    return params.spread_style;
  };
}

template <typename Uint>
  requires std::unsigned_integral<Uint>
void PrintBufferSelect(Uint value, const UnpackExactSelectedParams& params,
                       std::ostream& out = std::cout) {
  return PrintBytes(value, {
                               .bit_style = MakeBufferSelectColorFn(params),
                               .max_bytes = params.max_spread_bytes,
                           });
}

inline auto MakeShiftedBufferColorFn(const UnpackExactSelectedParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    // We look up the value as it would be in the full buffer
    const auto bit_off_idx = bit_idx + params.ValueBitStart();
    if (bit_idx < params.packed_bit_size) {
      const auto style_idx = params.value_idx % params.highlight_styles.size();
      return params.highlight_styles[style_idx];
    }
    // Values were out of input buffer so we show them as padded zeros
    if (bit_off_idx >= kBitsPerBytes * params.NBytes()) {
      return {};
    }
    if (bit_idx < params.MaxSpreadBits() - params.SpreadShiftBits()) {
      return params.spread_style;
    }
    // Color nothing for shifted zeros
    return {};
  };
}

template <typename Uint>
  requires std::unsigned_integral<Uint>
void PrintShiftedBuffer(Uint value, const UnpackExactSelectedParams& params,
                        std::ostream& out = std::cout) {
  return PrintBytes(value, {
                               .bit_style = MakeShiftedBufferColorFn(params),
                               .max_bytes = params.max_spread_bytes,
                           });
}

inline auto MakeMaskColorFn(const UnpackExactSelectedParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    if (bit_idx < params.packed_bit_size) {
      return params.mask_style;
    }
    return {};
  };
}

template <typename Uint>
  requires std::unsigned_integral<Uint>
void PrintMask(Uint value, const UnpackExactSelectedParams& params,
               std::ostream& out = std::cout) {
  return PrintBytes(value, {
                               .bit_style = MakeMaskColorFn(params),
                               .max_bytes = params.max_spread_bytes,
                           });
}

inline auto MakeUnpackedSelectColorFn(const UnpackExactSelectedParams& params) {
  return [=](std::size_t bit_idx) -> TextStyle {
    if (bit_idx < params.packed_bit_size) {
      const auto style_idx = params.value_idx % params.highlight_styles.size();
      return params.highlight_styles[style_idx];
    }
    return {};
  };
}

template <typename Uint>
  requires std::unsigned_integral<Uint>
void PrintUnpackedSelect(Uint value, const UnpackExactSelectedParams& params,
                         std::ostream& out = std::cout) {
  return PrintBytes(value, {
                               .bit_style = MakeUnpackedSelectColorFn(params),
                               .max_bytes = params.max_spread_bytes,
                           });
}
template <std::ranges::input_range Range, bool kUint8AsInt = true>
void PrintJoinedValues(const Range& values, std::string_view sep = ", ",
                       std::ostream& out = std::cout) {
  auto it = std::ranges::begin(values);
  auto end = std::ranges::end(values);

  if (it == end) {
    return;
  }

  const auto cast = []<typename V>(const V& v) {
    if constexpr (kUint8AsInt && std::is_same_v<V, std::uint8_t>) {
      return static_cast<int>(v);
    } else {
      return v;
    }
  };

  out << cast(*it);
  ++it;
  for (; it != end; ++it) {
    out << ", " << cast(*it);
  }
}

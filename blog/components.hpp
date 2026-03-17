#pragma once

#include <concepts>
#include <cstdint>
#include <iostream>
#include <utility>

#include <xcpp/xdisplay.hpp>
#include <xwidgets/xall.hpp>

#include "./utils.hpp"

template <std::unsigned_integral UintType = std::uint64_t>
class IntegerToolbar {
 public:
  using value_type = UintType;

  struct InitParams {
    std::string_view number_description = "Integer value";
    value_type number_min = 0;
    value_type number_default = 13;
    std::string_view uint_description = "Integer type";
  };

  IntegerToolbar(const InitParams& params = {}) {
    m_uint = xw::dropdown(
        std::vector<std::string>({"uint8_t", "uint16_t", "uint32_t", "uint64_t"}),
        "uint32_t");
    m_uint.description = params.uint_description;
    m_uint.value = "uint32_t";
    m_uint.style().description_width = "120px";
    m_number.min = params.number_min;
    m_number.value = params.number_default;
    m_number.description = params.number_description;
    m_number.style().description_width = "120px";

    m_toolbar.add(m_number);
    m_toolbar.add(m_uint);

    XOBSERVE(m_uint, value, [&](const auto& s) {
      const auto uint = ParseUint(s.value());
      const std::uint64_t uint_max = UintMax(uint);
      m_number.value = std::clamp(m_number.value(), std::uint64_t{0}, uint_max);
      m_number.max = uint_max;
    });
  }

  template <typename Func>
  void on_change(Func func) {
    auto callback = [this, func = std::move(func)](const auto& w) {
      auto g = m_output.guard();
      xcpp::clear_output(true);
      const auto uint = ParseUint(m_uint.value());
      func(w.value(), uint);
      std::cout << std::flush;
    };
    XOBSERVE(m_number, value, std::move(callback));
  }

  void display() {
    xcpp::display(m_toolbar);
    xcpp::display(m_output);
    m_number.value = m_number.value;  // Force first render
  }

 private:
  xw::dropdown m_uint = {};
  xw::number_bounded<value_type> m_number = {};
  xw::box m_toolbar = {};
  xw::output m_output = {};
};

template <std::unsigned_integral UintType = std::uint64_t>
class SequenceToolbar {
 public:
  using value_type = UintType;

  struct InitParams {
    std::size_t interval_ms = 500;
  };

  explicit SequenceToolbar(const InitParams& p = {}) {
    m_output.layout().min_height = "220px";
    m_csv_sequence.description = "Number sequence";
    m_csv_sequence.style().description_width = "120px";
    m_csv_sequence.value = "0, 1, 2, 3, 4, 5, 6, 7";
    m_player.interval = p.interval_ms;
    m_unpacked_uint = xw::dropdown(
        std::vector<std::string>({"uint8_t", "uint16_t", "uint32_t", "uint64_t"}),
        "uint32_t");
    m_unpacked_uint.description = "Integer type";
    m_unpacked_uint.style().description_width = "120px";
    m_unpacked_uint.value = "uint32_t";
    m_packed_bit_size.min = 0;
    m_packed_bit_size.max = unpacked_uint_max_bits();
    m_packed_bit_size.value = best_packed_bit_size();
    m_packed_bit_size.description = "Packed size (bits)";
    m_packed_bit_size.style().description_width = "120px";

    m_toolbar_params.add(m_unpacked_uint);
    m_toolbar_params.add(m_packed_bit_size);
    m_toolbar_seq.add(m_csv_sequence);
    m_toolbar_seq.add(m_player);

    auto set_bit_width_reset_player = [this](auto&&...) {
      m_player.value = 0;
      m_player.playing = true;
      try {
        m_packed_bit_size.value = best_packed_bit_size();
      } catch (const std::exception& e) {
      }
    };
    m_csv_sequence.on_submit(std::move(set_bit_width_reset_player));

    auto set_bit_width_pause_player = [this](auto&&...) {
      m_player.playing = false;
      try {
        m_packed_bit_size.value = best_packed_bit_size();
      } catch (const std::exception& e) {
      }
    };
    XOBSERVE(m_csv_sequence, value, std::move(set_bit_width_pause_player));
  }

  auto unpacked_uint() const -> Uint { return ParseUint(m_unpacked_uint.value()); }

  auto unpacked_uint_max_bits() const -> value_type {
    return static_cast<value_type>(unpacked_uint());
  }

  auto packed_bit_size() const -> value_type { return m_packed_bit_size.value(); }

  auto best_packed_bit_size() const -> value_type {
    return MaxWidthBits<value_type>(values());
  }

  auto values() const -> std::vector<value_type> {
    return ParseCsvInt<value_type>(m_csv_sequence.value());
  }

  auto step() const -> std::size_t { return m_player.value(); }

  template <typename Func>
  void on_change(Func func) {
    auto callback = [this, func = std::move(func)](const auto& w) {
      auto g = m_output.guard();
      xcpp::clear_output(true);
      func(std::as_const(*this));
      std::cout << std::flush;
    };
    XOBSERVE(m_player, value, std::move(callback));
  }

  void display() {
    xcpp::display(m_toolbar_params);
    xcpp::display(m_toolbar_seq);
    xcpp::display(m_output);
    m_player.value = m_player.value;  // Force first render
  }

 private:
  xw::dropdown m_unpacked_uint = {};
  xw::number_bounded<value_type> m_packed_bit_size = {};
  xw::text m_csv_sequence = {};
  xw::play m_player = {};
  xw::box m_toolbar_params = {};
  xw::box m_toolbar_seq = {};
  xw::output m_output = {};
};

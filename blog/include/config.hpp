#pragma once

#include <stdlib.h>
#include <format>
#include <string>

#include <clang/Interpreter/CppInterOp.h>

#ifndef __EMSCRIPTEN__

auto get_conda_prefix() -> const char* {
  static const char* CONDA_PREFIX = getenv("CONDA_PREFIX");
  return CONDA_PREFIX;
}

auto get_ld_library_path() -> std::string& {
  static std::string ENV_LD_LIBRARY_PATH = {};
  return ENV_LD_LIBRARY_PATH;
}

const bool INIT_LOCAL = []() {
  // Fix for LD_LIBRARY_PATH hard-coded in kernel.json
  if (auto conda_prefix = get_conda_prefix()) {
    auto& ld_library_path = get_ld_library_path();
    ld_library_path = std::format("LD_LIBRARY_PATH={}/lib/", conda_prefix);
    putenv(ld_library_path.data());
  }

  // Required libraries to link
  Cpp::LoadLibrary("libxwidgets");

  return true;
}();

// Fix for ARM ressources not being properly found
#  define __ARM_NEON_H
#  include "/Users/antoine/workspace/github.com/apache/arrow/.pixi/envs/jupyter/lib/clang/20/include/arm_bf16.h"
#  include "/Users/antoine/workspace/github.com/apache/arrow/.pixi/envs/jupyter/lib/clang/20/include/arm_vector_types.h"
#  undef __ARM_NEON_H
#  include "/Users/antoine/workspace/github.com/apache/arrow/.pixi/envs/jupyter/lib/clang/20/include/arm_neon.h"

#endif  // ! __EMSCRIPTEN__

const bool INIT_ALL = []() {
  // Arrow internal code
  Cpp::AddIncludePath("../cpp/src/");
  Cpp::AddIncludePath("./include/");

  return true;
}();

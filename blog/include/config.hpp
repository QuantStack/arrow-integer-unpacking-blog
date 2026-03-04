#pragma once

#include <clang/Interpreter/CppInterOp.h>

const bool INIT_ALL = []() {
  // Arrow internal code
  Cpp::AddIncludePath("../");
  Cpp::AddIncludePath("./include/");

  return true;
}();

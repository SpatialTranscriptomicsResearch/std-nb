#ifndef JIT_HPP
#define JIT_HPP

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "KaleidoscopeJIT.hpp"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

namespace JIT {

// LLVM
//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

void init_runtime(const std::string &module_name);
void define_log_exp();
void finalize_module(const std::string &Name);
std::function<double(const double *)> get_function(const std::string &Name);

struct Runtime {
  static llvm::LLVMContext TheContext;
  static llvm::IRBuilder<> Builder;
  static std::unique_ptr<llvm::Module> TheModule;
  static std::map<std::string, llvm::Value *> NamedValues;
  static std::unique_ptr<llvm::orc::KaleidoscopeJIT> TheJIT;

  static void InitializeModule(const std::string &module_name);
  static llvm::AllocaInst *CreateEntryBlockAlloca(llvm::Function *TheFunction,
                                                  const std::string &VarName,
                                                  llvm::Type *type);

  static llvm::Value *LogErrorV(const char *Str);
};

}  // namespace JIT

#endif

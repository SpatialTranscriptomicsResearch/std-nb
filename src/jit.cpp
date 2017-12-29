#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "jit.hpp"
#include "log.hpp"

using namespace llvm;

LLVMContext JIT::Runtime::TheContext;
IRBuilder<> JIT::Runtime::Builder(JIT::Runtime::TheContext);
std::unique_ptr<Module> JIT::Runtime::TheModule = nullptr;
std::map<std::string, Value *> JIT::Runtime::NamedValues;
std::unique_ptr<orc::KaleidoscopeJIT> JIT::Runtime::TheJIT = nullptr;

namespace JIT {
void init_runtime(const std::string &module_name, const std::string prefix) {
  LOG(debug) << "Running Init()";
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  Runtime::TheJIT = llvm::make_unique<orc::KaleidoscopeJIT>(prefix);

  Runtime::InitializeModule(module_name);

  JIT::define_log_exp();

  LOG(debug) << "Finished Init()";
}

Value *Runtime::LogErrorV(const char *Str) {
  LOG(warning) << Str;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

void Runtime::InitializeModule(const std::string &name) {
  // Open a new module.
  TheModule = llvm::make_unique<Module>(name, TheContext);
  TheModule->setTargetTriple(sys::getProcessTriple());
  TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
AllocaInst *Runtime::CreateEntryBlockAlloca(Function *function,
                                            const std::string &var_name,
                                            Type *type) {
  BasicBlock &block = function->getEntryBlock();
  IRBuilder<> tmp(&block, block.begin());
  return tmp.CreateAlloca(type, nullptr, var_name);
}

void define_log_exp() {
  using namespace llvm;
  Type *Double = llvm::Type::getDoubleTy(Runtime::TheContext);

  FunctionType *functype_ff = FunctionType::get(Double, Double, false);
  Function::Create(functype_ff, Function::ExternalLinkage, "exp",
                   Runtime::TheModule.get());
  Function::Create(functype_ff, Function::ExternalLinkage, "log",
                   Runtime::TheModule.get());
}

void finalize_module(const std::string &Name) {
  LOG(debug) << "Finalizing module";
  // Runtime::TheModule->dump();
  std::error_code EC;
  llvm::raw_fd_ostream ofs(Runtime::TheJIT->prefix + Name + ".ll", EC,
                           sys::fs::F_None);
  Runtime::TheModule->print(ofs, nullptr);
  ofs << "\n";

  Runtime::TheJIT->addModule(std::move(Runtime::TheModule));
}

std::function<double(const double *)> get_function(const std::string &Name) {
  // Search the JIT for the created symbol.
  auto ExprSymbol = Runtime::TheJIT->findSymbol(Name);
  assert(ExprSymbol && "Function not found");

  // Get the symbol's address and cast it to the right type (takes a
  // double * argument, returns a double) so we can call it as a native
  // function.
  double (*FP)(const double *)
      = (double (*)(const double *))cantFail(ExprSymbol.getAddress());
  // TODO handle failure

  LOG(debug) << "getAdress() = "
               << (intptr_t)cantFail(ExprSymbol.getAddress());

  return FP;
}
}  // namespace JIT

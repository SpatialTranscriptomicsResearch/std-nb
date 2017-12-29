//===- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "log.hpp"

namespace llvm {
namespace orc {

class KaleidoscopeJIT {
public:
  std::string prefix;
private:
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

  using OptimizeFunction
      = std::function<std::shared_ptr<Module>(std::shared_ptr<Module>)>;

  IRTransformLayer<decltype(CompileLayer), OptimizeFunction> OptimizeLayer;

public:
  using ModuleHandle = decltype(OptimizeLayer)::ModuleHandleT;

  KaleidoscopeJIT(const std::string &prefix_)
      : prefix(prefix_),
        TM(EngineBuilder()
               .setMCPU(llvm::sys::getHostCPUName())
               .setEngineKind(llvm::EngineKind::JIT)
               .setOptLevel(llvm::CodeGenOpt::Aggressive)
               .selectTarget()),
        DL(TM->createDataLayout()),
        ObjectLayer([]() { return std::make_shared<SectionMemoryManager>(); }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        OptimizeLayer(CompileLayer, [this](std::shared_ptr<Module> M) {
          return optimizeModule(std::move(M));
        }) {
    LOG(debug) << "Constructing KaleidoscopeJIT";
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  TargetMachine &getTargetMachine() { return *TM; }

  ModuleHandle addModule(std::unique_ptr<Module> M) {
    // Build our symbol resolver:
    // Lambda 1: Look back into the JIT itself to find symbols that are part of
    //           the same "logical dylib".
    // Lambda 2: Search for external symbols in the host process.
    auto Resolver = createLambdaResolver(
        [&](const std::string &Name) {
          if (auto Sym = OptimizeLayer.findSymbol(Name, false))
            return Sym;
          return JITSymbol(nullptr);
        },
        [](const std::string &Name) {
          if (auto SymAddr
              = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
            return JITSymbol(SymAddr, JITSymbolFlags::Exported);
          return JITSymbol(nullptr);
        });

    // Add the set to the JIT with the resolver we created above and a newly
    // created SectionMemoryManager.
    return cantFail(OptimizeLayer.addModule(std::move(M), std::move(Resolver)));
  }

  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return OptimizeLayer.findSymbol(MangledNameStream.str(), true);
  }

  void removeModule(ModuleHandle H) { cantFail(OptimizeLayer.removeModule(H)); }

private:
  std::shared_ptr<Module> optimizeModule(std::shared_ptr<Module> M) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(M.get());

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->add(createInstructionSimplifierPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *M)
      FPM->run(F);

    LOG(verbose) << "Optimized module";
    std::error_code EC;
    std::string path = prefix + M->getModuleIdentifier() + "-optimized.ll";
    llvm::raw_fd_ostream ofs(path, EC, sys::fs::F_None);
    M->print(ofs, nullptr);
    ofs << "\n";
    // M->dump();

    return M;
  }
};

}  // end namespace orc
}  // end namespace llvm

#endif  // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

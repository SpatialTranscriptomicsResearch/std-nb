#ifndef GENERATE_HPP
#define GENERATE_HPP

#include "collect.hpp"
#include "exp.hpp"

namespace spec_parser {
namespace expression {

// TODO move out of header
// call JIT::Runtime::InitializeModule before using this function
// call JIT::Runtime::define_log_exp before using this function
template <typename T>
void codegen(const ExpPtr<T> &expr, const std::string &Name) {
  LOG(verbose) << "Generating code '" << Name << "' for expression "
               << show(expr);
  using namespace llvm;

  // TODO: consider function type: void(double*,double*) etc.
  // Type *Int = llvm::Type::getInt64Ty(JIT::Runtime::TheContext);
  Type *Double = llvm::Type::getDoubleTy(JIT::Runtime::TheContext);
  Type *DblPtr = llvm::PointerType::get(Double, 0);

  FunctionType *FT = FunctionType::get(Double, DblPtr, false);
  Function *TheFunction = Function::Create(FT, Function::ExternalLinkage, Name,
                                           JIT::Runtime::TheModule.get());

  // Get pointer to the double* argument of the function...
  assert(TheFunction->arg_begin()
         != TheFunction->arg_end());            // Make sure there's an arg
  Argument *ArgX = &*TheFunction->arg_begin();  // Get the arg
  ArgX->setName("x");  // Give it a nice symbolic name for fun.

  // Create a new basic block to start insertion into.
  BasicBlock *BB
      = BasicBlock::Create(JIT::Runtime::TheContext, "entry", TheFunction);

  JIT::Runtime::Builder.SetInsertPoint(BB);
  JIT::Runtime::NamedValues.clear();

  // TODO ensure the correct order is used
  auto variables = collect_variables(expr);

  size_t idx = 0;
  for (auto variable : variables) {
    std::string name = to_string(*variable);
    Type *Int64 = IntegerType::get(JIT::Runtime::TheContext, 64);
    Value *offset = ConstantInt::get(Int64, idx++);
    Value *Addr = JIT::Runtime::Builder.CreateGEP(ArgX, {offset}, "addr");
    Value *value = JIT::Runtime::Builder.CreateLoad(Addr, name.c_str());
    JIT::Runtime::NamedValues[name] = value;
  }

  if (Value *RetVal = expr->codegen()) {
    // Finish off the function.
    JIT::Runtime::Builder.CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    if (verifyFunction(*TheFunction, &llvm::errs()))
      throw std::runtime_error("An error occurred while generation code '"
                               + Name + "' for the expression " + show(expr));

    LOG(verbose) << "Code generation succeeded for expression " << show(expr);
  } else {
    LOG(fatal) << "Code generation failed for expression " << show(expr);
    LOG(fatal) << "Removing function from parent.";
    // Error reading body, remove function.
    TheFunction->eraseFromParent();
  }
}

}  // namespace expression
}  // namespace spec_parser
#endif

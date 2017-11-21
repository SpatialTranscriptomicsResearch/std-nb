#ifndef SPECPARSER_EXPRESSION_CODEGEN_HPP
#define SPECPARSER_EXPRESSION_CODEGEN_HPP

#include <memory>
#include "jit.hpp"
// #include "log.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
Value* Num<T>::codegen() {
  // LOG(verbose) << "codegen: value " << value;
  return llvm::ConstantFP::get(JIT::Runtime::TheContext, llvm::APFloat(value));
}

template <typename T>
Value* Var<T>::codegen() {
  // LOG(verbose) << "codegen: variable " << value;
  std::string name = to_string(*value);

  // Look this variable up in the function.
  Value* V = JIT::Runtime::NamedValues[name];
  if (!V)
    return JIT::Runtime::LogErrorV("Unknown variable name");
  return V;
}

template <typename T>
Value* Unr<T>::codegen() {
  // LOG(verbose) << "codegen: unr";
  Value* op_value = operand->codegen();
  if (!op_value)
    return nullptr;
  // LOG(verbose) << "codegen: unr op_value";
  // op_value->print(llvm::errs());

  switch (op) {
    case Type::NEG: {
      return JIT::Runtime::Builder.CreateFNeg(op_value, "negtmp");
    } break;
    case Type::INV: {
      Value* one
          = llvm::ConstantFP::get(JIT::Runtime::TheContext, llvm::APFloat(1.0));
      return JIT::Runtime::Builder.CreateFDiv(one, op_value, "invtmp");
    } break;
    case Type::EXP: {
      llvm::Function* F = JIT::Runtime::TheModule->getFunction("exp");
      std::vector<Value*> args = {op_value};
      return JIT::Runtime::Builder.CreateCall(F, args, "exptmp");
    }
    case Type::LOG: {
      llvm::Function* F = JIT::Runtime::TheModule->getFunction("log");
      std::vector<Value*> args = {op_value};
      return JIT::Runtime::Builder.CreateCall(F, args, "logtmp");
    } break;
  }
  return nullptr;
}

template <typename T>
Value* Bin<T>::codegen() {
  // LOG(verbose) << "codegen: bin";
  Value* l = lhs->codegen();
  Value* r = rhs->codegen();
  if (!l || !r)
    return nullptr;
  // LOG(verbose) << "codegen: bin l";
  // l->print(llvm::errs());
  // LOG(verbose) << "codegen: bin r";
  // r->print(llvm::errs());

  switch (op) {
    case Type::ADD:
      return JIT::Runtime::Builder.CreateFAdd(l, r, "addtmp");
    case Type::MUL:
      return JIT::Runtime::Builder.CreateFMul(l, r, "multmp");
  }
  return nullptr;
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_CODEGEN_HPP

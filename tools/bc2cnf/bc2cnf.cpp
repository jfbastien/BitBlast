//===-- bc2cnf.cpp - .bc -> .cnf ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InstVisitor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ToolOutputFile.h"
#include <vector>

using namespace llvm;

namespace {

const char Bc2CNFDescription[] =
  "llvm .bc -> .cnf\n"
  "\n"
  "  Bitcode file (.bc) to DIMACS Conjunctive Normal Form file (.cnf).\n"
  "\n"
  "  Format:\n"
  "   - Initial comment lines starting with 'c'.\n"
  "   - A line with: 'p cnf <nbvars> <nbclauses>'.\n"
  "   - One line per clause, with:\n"
  "     - Sequence of non-null numbers between -<nbvars> and <nbvars>.\n"
  "     - Ending with 0 on the same line.\n"
  "     - Clauses can't contain i and -i simultaneously.\n"
  "     - Positive numbers denote the corresponding variable.\n"
  "     - Negative numbers denote their negation.\n";

cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

cl::opt<std::string>
OutputFilename("o", cl::desc("<output cnf>"), cl::value_desc("filename"));

struct CNFVisitor : public InstVisitor<CNFVisitor> {
  typedef int64_t Variable;
  typedef std::vector<Variable> Clause;
  typedef std::vector<Clause> Formula;

  Variable Vars;
  Formula Fa;
  DenseMap<Value *, Variable> Map;
  void Set(Value *Val, Variable Var) {
    assert(Map.find(Val) == Map.end());
    Map[Val] = Var;
  }
  Variable Get(Value *Val) {
    assert(Map.find(Val) != Map.end());
    return Map[Val];
  }

  CNFVisitor() : Vars(0) {}

  void visitExtractElementInst(ExtractElementInst &I) {
    // An input variable, assumed to be in sorted bit order.
    Set(&I, ++Vars);
  }
  void visitOr(BinaryOperator &I) {
    Set(&I, ++Vars);
  }
  void visitXor(BinaryOperator &I) {
    // NOT is implemented as XOR B, 1.
    assert(isa<ConstantInt>(I.getOperand(1)));
    assert(cast<ConstantInt>(I.getOperand(1))->
           getUniqueInteger().getLimitedValue() == 1);
    Set(&I, -Get(I.getOperand(0)));
  }
  void visitInsertElementInst(InsertElementInst &I) {
    // An output variable.
  }
  void visitReturnInst(ReturnInst &I) {
    // Mandatory terminator.
  }
  void visitAnd(BinaryOperator &I) {
    // Recursively traverse all OR/NOT instructions that make up this clause.
    Clause C;
    traverse(C, I.getOperand(0));
    traverse(C, I.getOperand(1));
    Fa.push_back(C);
    Set(&I, ++Vars);
  }
  // Fallthrough instruction visitor.
  void visitInstruction(Instruction &I) {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << "Unimplemented instruction: " << I;
    report_fatal_error(OS.str());
  }


  void traverse(Clause &C, Value *V) {
    Instruction *I = dyn_cast<Instruction>(V);
    assert(I);
    switch (I->getOpcode()) {
      case Instruction::Or:
        // Traverse all OR variables' constituents recursively.
        traverse(C, I->getOperand(0));
        traverse(C, I->getOperand(1));
      return;
      case Instruction::And:
      case Instruction::Xor:
      case Instruction::ExtractElement:
        // Non-OR instructions are terminators.
        C.push_back(Get(V));
        return;
      default: {
        std::string Str;
        raw_string_ostream OS(Str);
        OS << "Unimplemented instruction: " << *I;
        report_fatal_error(OS.str());
      }
    }
  }

};


void Function2CNF(formatted_raw_ostream &o, Function &F) {
  CNFVisitor V;
  V.visit(F);

  o << "c " << F.getName() << '\n' <<
      "p cnf " << V.Fa.size() << ' ' << V.Vars << '\n';

  for (CNFVisitor::Formula::const_iterator FI(V.Fa.begin());
       FI != V.Fa.end(); ++FI) {
    for (CNFVisitor::Clause::const_iterator CI(FI->begin());
         CI != FI->end(); ++CI)
      o << *CI << ' ';
    o << "0\n";
  }
}

int Bc2CNF(StringRef ProgramName, LLVMContext &Context) {
  SMDiagnostic Err;
  OwningPtr<Module> M;

  // Parse input.
  M.reset(ParseIRFile(InputFilename, Err, Context));
  if (M.get() == 0) {
    Err.print(ProgramName.data(), errs());
    return 1;
  }

  for (Module::iterator F(M->begin()); F != M->end(); ++F) {

    // Setup output.
    OwningPtr<tool_output_file> Out;
    {
      std::string ErrorInfo;
      Out.reset(new tool_output_file(
          (OutputFilename + "_" + F->getName()).str().c_str(),
          ErrorInfo, sys::fs::F_Binary));
      if (!ErrorInfo.empty()) {
        errs() << ErrorInfo << '\n';
        return 1;
      }
    }
    formatted_raw_ostream FOS(Out->os());
    Function2CNF(FOS, *F);
    Out->keep();
  }

  return 0;
}

}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  EnableDebugBuffering = true;
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, Bc2CNFDescription);
  return Bc2CNF(argv[0], getGlobalContext());
}

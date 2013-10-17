//===- BitBlast.cpp - Transform functions to boolean formulas -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Function pass transforming each function into AND, OR and NOT
// operations on boolean variables. Function arguments and return value
// are transformed into bit inputs and outputs respectively.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "bitblast"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include <algorithm>
#include <vector>
using namespace llvm;

namespace {

cl::opt<bool>
OptMemoize("BitBlast-memoize",
           cl::desc("Memoize common bit expressions"),
           cl::init(true));
cl::opt<bool>
OptConstProp("BitBlast-constprop",
             cl::desc("Perform bit constant propagation"),
             cl::init(true));
cl::opt<bool>
OptFancyNames("BitBlast-fancy-names",
              cl::desc("Use fancy variable names (bad idea: uses memory)"),
              cl::init(false));

// Bit maps use SmallVector, which has a compile-time initial allocation
// size. Most values should be 32 or 64 bits, one of these values should
// therefore be used.
static const size_t BitMapInitialSize = 32;

// Generic helpers.
size_t TypeBitWidth(Type *T) {
  size_t S = 0;
  switch (T->getTypeID()) {
    case Type::IntegerTyID: S = T->getIntegerBitWidth(); break;
    case Type::StructTyID:
      // We could calculate this by recursing, but we instead rely on
      // SROA to take care of aggregate types: it simplifies other
      // parts of bit blasting.
    default: {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unimplemented type: " << *T;
      report_fatal_error(OS.str());
    } break;
  }
  return S;
}
size_t TypeBitWidth(Value *V) { return TypeBitWidth(V->getType()); }
size_t TypeBitWidth(Value &V) { return TypeBitWidth(V.getType()); }

struct BitBlastVisitor : public InstVisitor<BitBlastVisitor> {
  Module &M;
  LLVMContext &C;
  Type *Bool; // Boolean type.
  Type *I32; // 32-bit integer type.
  Function *F; // New boolean Function.
  OwningPtr<IRBuilder<> > B;
  size_t ReturnsSeen; // Lazy: we don't handle multiple returns.

  BitBlastVisitor(Module &M, Function *FOld)
      : M(M),
        C(M.getContext()),
        Bool(Type::getInt1Ty(C)),
        I32(Type::getInt32Ty(C)),
        F(NULL),
        ReturnsSeen(0)
  {
    // Create a replacement function which will have a single basic
    // block and input/output boolean vectors of the same bit width.
    F = Function::Create(BooleanFunctionType(FOld), FOld->getLinkage(),
                         "boolean_" + FOld->getName(), &M);
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    B.reset(new IRBuilder<>(C));
    B->SetInsertPoint(BB);
    MapInputBits(FOld);
  }

  // Boolean function type signature with bit vectors as input and
  // output, whose width corresponds to the bit width of the old
  // function's inputs and outputs.
  FunctionType *BooleanFunctionType(Function *FOld) {
    Type *RT = VectorType::get(Bool, TypeBitWidth(FOld->getReturnType()));
    FunctionType *OldFT = FOld->getFunctionType();
    if (OldFT->isVarArg()) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unsupported vararg function: " << FOld->getName();
      report_fatal_error(OS.str());
    }
    size_t ParamBits = 0;
    for (size_t P(0); P != OldFT->getNumParams(); ++P)
      ParamBits += TypeBitWidth(OldFT->getParamType(P));
    Type *Params[1] = { VectorType::get(Bool, ParamBits) };
    return FunctionType::get(RT, Params, false);
  }

  // Create boolean values in the boolean function corresponding to each
  // input argument bit in the original function (represented by the
  // boolean function's input vector of the corresponding width).
  void MapInputBits(Function *FOld) {
    size_t InputBits = 0;
    Value *InputVector = &*F->arg_begin();
    for (Function::arg_iterator A(FOld->arg_begin()); A != FOld->arg_end();
         ++A) {
      Value *OldValue = &*A;
      BitMap &Input(NewBitMap(OldValue));
      for (size_t OldBit = 0, OldBitWidth = TypeBitWidth(OldValue);
           OldBit != OldBitWidth; ++OldBit) {
        Value *Idx = ConstantInt::get(I32, InputBits);
        std::string Str;
        raw_string_ostream OS(Str);
        OS << "in" << InputBits;
        Value *Bit = B->CreateExtractElement(InputVector, Idx, OS.str());
        Input.push_back(Bit);
        ++InputBits;
      }
    }
  }

  // Map Values in the old functions to the N equivalent bit values in
  // the new boolean functions. The N bits are stored with the LSB at
  // position 0.
  typedef SmallVector<Value *, BitMapInitialSize> BitMap;
  DenseMap<Value *, BitMap> V2B;
  // Add a value from the old function to the value -> N bit values map,
  // and return an empty vector meant to hold the N bit values in the
  // new function.
  BitMap &NewBitMap(Value *OldValue) {
    assert(V2B.find(OldValue) == V2B.end() && "Value already exists");
    V2B[OldValue] = BitMap();
    return V2B[OldValue];
  }
  BitMap &NewBitMap(Value &OldValue) { return NewBitMap(&OldValue); }
  // Get the Nth bit value in the new function corresponding to the old
  // function's value's Nth bit, or the boolean constant corresponding
  // to the original value.
  Value *GetBit(Value *OldValue, size_t Bit) const {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(OldValue)) {
      const APInt &API(CI->getUniqueInteger());
      bool BoolValue = (((API.lshr(Bit)) & APInt(API.getBitWidth(), 1)).
                        getLimitedValue());
      return ConstantBool(BoolValue);
    } else {
      assert(!isa<Constant>(OldValue) && "Non-integral constant value");
      assert(V2B.find(OldValue) != V2B.end() && "Value doesn't exist");
      assert(Bit < V2B.find(OldValue)->second.size() && "Size is wrong");
      return V2B.find(OldValue)->second[Bit];
    }
  }
  Value *GetBit(Value &OldValue, size_t Bit) const {
    return GetBit(&OldValue, Bit);
  }

  // Used in constant-propagation. Note that the zero value ("not a
  // constant bool") is used as ``false`` in code that uses this type to
  // test whether a value is constant or not (i.e. enum order matters).
  enum MaybeConstantBool { NotConstantBool, BoolTrue, BoolFalse };
  static MaybeConstantBool IsConstantBool(Value *V) {
    if (!OptConstProp)
      return NotConstantBool;
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return CI->getUniqueInteger().getLimitedValue() ? BoolTrue : BoolFalse;
    return NotConstantBool;
  }

  // Used in binary operations which need a constant integral right hand
  // side operand. Error out when not constant or integral.
  static uint64_t GetConstantRHS(Instruction &I) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(1)))
      return CI->getUniqueInteger().getLimitedValue();
    std::string Str;
    raw_string_ostream OS(Str);
    OS << "Unsupported RHS (must be integer constant): " << I;
    report_fatal_error(OS.str());
  }

  // The following are used for eager CSE at build time. Note that we
  // don't memoize all potential expressions: higher-level CSE is
  // expected to have been performed before bit blasting. Only bit-wise
  // primitives are memoized.
  typedef DenseMap<Value *, Value *> UnaryMemoize;
  typedef DenseMap<std::pair<Value *, Value *>, Value *> BinaryMemoize;
  UnaryMemoize NotMap;
  BinaryMemoize AndMap;
  BinaryMemoize OrMap;
  BinaryMemoize XorMap;
  BinaryMemoize XnorMap;
  Value *IsMemoized(const UnaryMemoize &M, Value *V) const {
    if (!OptMemoize)
      return NULL;
    UnaryMemoize::const_iterator Mem = M.find(V);
    return Mem != M.end() ? Mem->second : (Value *) NULL;
  }
  Value *IsMemoized(const BinaryMemoize &M, Value *L, Value *R) const {
    if (!OptMemoize)
      return NULL;
    BinaryMemoize::const_iterator Mem = M.find(std::make_pair(L, R));
    return Mem != M.end() ? Mem->second : (Value *) NULL;
  }
  Value *Memoize(UnaryMemoize &M, Value *V, Value *Res) {
    if (!OptMemoize)
      return Res;
    return M[V] = Res;
  }
  Value *Memoize(BinaryMemoize &M, Value *L, Value *R, Value *Res) {
    if (!OptMemoize)
      return Res;
    // Insert both permutations so that lookups don't have to check both.
    return M[std::make_pair(L, R)] = M[std::make_pair(R, L)] = Res;
  }

  // Builder helpers. These only ever emit AND, OR and NOT, but they
  // also try to memoize low-level bit computations as well as perform
  // some constant propagation.
  Value *ConstantBool(bool BoolValue) const {
    return ConstantInt::get(Bool, BoolValue);
  }
  Value *And(Value *L, Value *R) {
    MaybeConstantBool CL = IsConstantBool(L), CR = IsConstantBool(R);
    if (CL && CR) return ConstantBool((CL == BoolTrue) & (CR == BoolTrue));
    if (CL) return CL == BoolTrue ? R : ConstantBool(0); // 1&R == R ; 0&R == 0
    if (CR) return CR == BoolTrue ? L : ConstantBool(0); // L&1 == L ; L&0 == 0
    if (Value *Mem = IsMemoized(AndMap, L, R)) return Mem;
    return Memoize(AndMap, L, R,
                   B->CreateAnd(
                       L, R, OptFancyNames ?
                       ("(" + L->getName() + "&" + R->getName() + ")") : ""));
  }
  Value *Or(Value *L, Value *R) {
    MaybeConstantBool CL = IsConstantBool(L), CR = IsConstantBool(R);
    if (CL && CR) return ConstantBool((CL == BoolTrue) | (CR == BoolTrue));
    if (CL) return CL == BoolTrue ? ConstantBool(1) : R; // 1|R == 1 ; 0|R == R
    if (CR) return CR == BoolTrue ? ConstantBool(1) : L; // L|1 == 1 ; L|0 == L
    if (Value *Mem = IsMemoized(OrMap, L, R)) return Mem;
    return Memoize(OrMap, L, R,
                   B->CreateOr(
                       L, R, OptFancyNames ?
                       ("(" + L->getName() + "|" + R->getName() + ")") : ""));
  }
  Value *Not(Value *V) {
    if (MaybeConstantBool C = IsConstantBool(V))
      return ConstantBool(C != BoolTrue); // ~1 == 0 ; ~0 == 1
    if (Value *Mem = IsMemoized(NotMap, V)) return Mem;
    return Memoize(NotMap, V,
                   B->CreateNot( V, OptFancyNames ? ("~" + V->getName()) : ""));
  }
  Value *Xor(Value *L, Value *R) {
    MaybeConstantBool CL = IsConstantBool(L), CR = IsConstantBool(R);
    if (CL && CR) return ConstantBool((CL == BoolTrue) ^ (CR == BoolTrue));
    if (CL) return CL == BoolTrue ? Not(R) : R; // 1^R == ~R ; 0^R == R
    if (CR) return CR == BoolTrue ? Not(L) : L; // L^1 == ~L ; L^0 == L
    if (Value *Mem = IsMemoized(XorMap, L, R)) return Mem;
    // L^R == (L|R)&~(L&R)
    // Note that the following is equivalent but produces ~30% more
    // code: (L&~R)|(~L&R)
    return Memoize(XorMap, L, R, And(Or(L, R), Not(And(L, R))));
  }
  Value *Xnor(Value *L, Value *R) {
    MaybeConstantBool CL = IsConstantBool(L), CR = IsConstantBool(R);
    if (CL && CR) return ConstantBool((CL == BoolTrue) == (CR == BoolTrue));
    if (CL) return CL == BoolTrue ? R : Not(R); // 1oR == R ; 0oR == ~R
    if (CR) return CR == BoolTrue ? L : Not(L); // Lo1 == L ; Lo0 == ~L
    if (Value *Mem = IsMemoized(XnorMap, L, R)) return Mem;
    // LoR == (L|R)&(~L&~R)
    return Memoize(XnorMap, L, R, And(Or(L, R), And(Not(L), Not(R))));
  }

  // Visitor helpers.

  // Add/subtract helper, which can also generate carry-out and
  // zero-out. The later two are useful for comparison.
  enum AddSubOpts { IsAddOpt, IsSubOpt, IsLtOpt, IsLeOpt };
  void visitAddSub(AddSubOpts Opts, Instruction &I, Value *L, Value *R) {
    bool IsAdd = Opts == IsAddOpt; // Comparisons are subtractions.
    bool IsCmp = (Opts == IsLtOpt) || (Opts == IsLeOpt);
    bool GenCarry = IsCmp;
    bool GenZero = Opts == IsLeOpt;
    bool GenRes = (GenZero || !IsCmp); // Don't generate a result if not needed.
    dbgs() << "Visit " << I << "\n";
    size_t BitWidth = TypeBitWidth(I);
    assert(TypeBitWidth(L) == TypeBitWidth(R) && "Inconsistent bit width");
    assert((BitWidth == (IsCmp ? 1 : TypeBitWidth(L))) &&
           "Inconsistent bit width");
    // Holds the result associated with the instruction.
    BitMap &Res(NewBitMap(I));
    BitMap ResForZero; // Holds the result when generating zero.
    BitMap *ResHolder(GenZero ? &ResForZero : (Opts == IsLtOpt ? NULL : &Res));
    assert((GenRes ? ResHolder != NULL : true) && "Need a result holder");
    // Ripple-carry adder/subtractor.
    // TODO Make this more efficient?
    // First bit is a half-adder/subtractor.
    Value *LB = GetBit(L, 0), *RB = GetBit(R, 0);
    if (GenRes)
      ResHolder->push_back(Xor(LB, RB));
    Value *Carry = And(IsAdd ? LB : Not(LB), RB);
    for (size_t Bit = 1; Bit < BitWidth; ++Bit) {
      // Other bits are full-adders/subtractors.
      LB = GetBit(L, Bit);
      RB = GetBit(R, Bit);
      Value *Xor1 = Xor(LB, RB);
      if (GenRes)
        ResHolder->push_back(Xor(Xor1, Carry));
      if (GenCarry || (Bit != BitWidth - 1))
        // Drop the carry on the last bit, unless doing a comparison.
        Carry = Or(And(IsAdd ? Xor1 : Not(Xor1), Carry),
                   And(IsAdd ? LB : Not(LB), RB));
    }
    Value *Zero = NULL;
    if (GenZero) {
      // TODO Add parallelism?
      // Equality compare all result bits with zero.
      assert(ResHolder == &ResForZero && "Result went to the wrong place");
      assert(ResForZero.size() == BitWidth && "Missing bits");
      for (size_t Bit = 0; Bit != BitWidth; ++Bit) {
        Value *BitNot = Not(ResForZero[Bit]);
        Zero = Bit == 0 ? BitNot : And(Zero, BitNot);
      }
    }
    if (Opts == IsLtOpt)
      // L < R if carry is set.
      Res.push_back(Carry);
    if (Opts == IsLeOpt)
      // L <= R if carry or zero are set.
      Res.push_back(Or(Carry, Zero));
    assert(Res.size() == BitWidth && "Missing bits");
  }
  // Straight binary operations which have an equivalent builder helper.
  void visitStraightBinary(Value *(BitBlastVisitor::*F)(Value *, Value *),
                           BinaryOperator &I) {
    dbgs() << "Visit " << I << "\n";
    size_t BitWidth = TypeBitWidth(I);
    Value *L = I.getOperand(0), *R = I.getOperand(1);
    assert(BitWidth == TypeBitWidth(L) && "Inconsistent bit width");
    assert(BitWidth == TypeBitWidth(R) && "Inconsistent bit width");
    BitMap &Res(NewBitMap(I));
    for (size_t Bit = 0; Bit != BitWidth; ++Bit)
      Res.push_back((this->*F)(GetBit(L, Bit), GetBit(R, Bit)));
    assert(Res.size() == BitWidth && "Straight binary badly implemented");
  }
  // Truncation and bitcast preserve bits the same way.
  void visitTruncCast(Instruction &I) {
    dbgs() << "Visit " << I << "\n";
    Value *V = I.getOperand(0);
    size_t BitWidthTo = TypeBitWidth(I),  BitWidthFrom = TypeBitWidth(V);
    assert(BitWidthTo <= BitWidthFrom && "Truncation/cast must narrow");
    BitMap &Res(NewBitMap(I));
    for (size_t Bit = 0; Bit != BitWidthTo; ++Bit)
      Res.push_back(GetBit(V, Bit));
    assert(Res.size() == BitWidthTo && "Truncation badly implemented");
  }
  // Zero- and sign-extension only differ in how they populate the top bits.
  void visitExt(bool IsSigned, Instruction &I) {
    dbgs() << "Visit " << I << "\n";
    Value *V = I.getOperand(0);
    size_t BitWidthTo = TypeBitWidth(I),  BitWidthFrom = TypeBitWidth(V);
    assert(BitWidthTo > BitWidthFrom && "Extension must widen");
    BitMap &Res(NewBitMap(I));
    Value *TopBit = IsSigned ? GetBit(V, BitWidthFrom - 1) : ConstantBool(0);
    for (size_t Bit = 0; Bit != BitWidthTo; ++Bit)
      Res.push_back(Bit < BitWidthFrom ? GetBit(V, Bit) : TopBit);
    assert(Res.size() == BitWidthTo && "Extension badly implemented");
  }
  // Select and PHI share a lot of code for final selection.
  void visitSelectPHI(Value *C, Value *T, Value *F, Instruction &I) {
    size_t BitWidth = TypeBitWidth(I);
    assert(1 == TypeBitWidth(C) && "Conditions have bit width of 1");
    assert(BitWidth == TypeBitWidth(T) && "Inconsistent bit width");
    assert(BitWidth == TypeBitWidth(F) && "Inconsistent bit width");
    Value *CB = GetBit(C, 0), *NB = Not(CB);
    BitMap &Res(NewBitMap(I));
    for (size_t Bit = 0; Bit != BitWidth; ++Bit) {
      Value *TB = GetBit(T, Bit), *FB = GetBit(F, Bit);
      Res.push_back(Or(And(CB, TB), And(NB, FB)));
    }
    assert(Res.size() == BitWidth && "Select/PHI badly implemented");
  }

  // Visitors.

  // Binary.
  void visitAdd(BinaryOperator &I) {
    visitAddSub(IsAddOpt, I, I.getOperand(0), I.getOperand(1));
  }
  void visitSub(BinaryOperator &I) {
    visitAddSub(IsSubOpt, I, I.getOperand(0), I.getOperand(1));
  }

  // Logical.
  void visitShl(BinaryOperator &I) {
    dbgs() << "Visit " << I << "\n";
    Value *L = I.getOperand(0);
    size_t BitWidth = TypeBitWidth(L);
    uint64_t Shift = GetConstantRHS(I); // TODO Handle UB Shift > BitWidth.
    BitMap &Res(NewBitMap(I));
    size_t DstBit = 0;
    for ( ; DstBit < Shift && DstBit < BitWidth; ++DstBit)
      Res.push_back(ConstantBool(0));
    for (size_t SrcBit = 0; DstBit < BitWidth; ++DstBit, ++SrcBit)
      Res.push_back(GetBit(L, SrcBit));
    assert(Res.size() == BitWidth && "Inconsistent bit width");
  }
  void visitLShr(BinaryOperator &I) { visitShr(false, I); }
  void visitAShr(BinaryOperator &I) { visitShr(true, I); }
  void visitShr(bool IsArithmetic, BinaryOperator &I) {
    dbgs() << "Visit " << I << "\n";
    Value *L = I.getOperand(0);
    size_t BitWidth = TypeBitWidth(L);
    uint64_t Shift = GetConstantRHS(I); // TODO Handle UB Shift > BitWidth.
    Value *TopBit = IsArithmetic ? GetBit(L, BitWidth - 1) : ConstantBool(0);
    BitMap &Res(NewBitMap(I));
    size_t DstBit = 0;
    for ( ; DstBit < BitWidth; ++DstBit)
      Res.push_back((DstBit < BitWidth - Shift) ?
                    GetBit(L, Shift + DstBit) : TopBit);
    assert(Res.size() == BitWidth && "Inconsistent bit width");
  }
  void visitAnd(BinaryOperator &I) {
    visitStraightBinary(&BitBlastVisitor::And, I);
  }
  void visitOr(BinaryOperator &I) {
    visitStraightBinary(&BitBlastVisitor::Or, I);
  }
  void visitXor(BinaryOperator &I) {
    visitStraightBinary(&BitBlastVisitor::Xor, I);
  }

  // Other.
  void visitReturnInst(ReturnInst &I) {
    dbgs() << "Visit " << I << "\n";
    if (ReturnsSeen++)
      report_fatal_error("Unimplemented: more than one return in a function");
    Value *V = I.getOperand(0);
    size_t BitWidth = TypeBitWidth(V);
    Value *Vector = UndefValue::get(VectorType::get(Bool, BitWidth));
    for (size_t Bit = 0; Bit != BitWidth; ++Bit) {
      Value *Idx = ConstantInt::get(I32, Bit);
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "out" << Bit;
      Vector = B->CreateInsertElement(Vector, GetBit(V, Bit), Idx, OS.str());
    }
    B->CreateRet(Vector);
  }
  void visitBranchInst(BranchInst &I) {
    // Ignore branches: we merge values at PHIs instead.
    // Backedges should fail their GetBit lookup.
    dbgs() << "Visit " << I << "\n";
  }
  void visitICmpInst(ICmpInst &I) {
    dbgs() << "Visit " << I << "\n";
    assert(TypeBitWidth(I) == 1 && "Non-predicate bit returning comparison");
    CmpInst::Predicate Pred = I.getPredicate();
    Value *L = I.getOperand(0), *R = I.getOperand(1);
    size_t BitWidth = TypeBitWidth(L);
    assert(BitWidth == TypeBitWidth(R) && "Inconsistent bit width");
    if (BitWidth == 1) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unsupported compare bitwidth of 1: " << I;
      report_fatal_error(OS.str());
    }
    switch (Pred) {
      case CmpInst::ICMP_EQ: {
        // TODO Add parallelism?
        Value *Prev;
        for (size_t Bit = 0; Bit != BitWidth; ++Bit) {
          Value *BitCompare = Xnor(GetBit(L, Bit), GetBit(R, Bit));
          Prev = Bit == 0 ? BitCompare : And(Prev, BitCompare);
        }
        NewBitMap(I).push_back(Prev);
        return;
      }
      case CmpInst::ICMP_NE: {
        // TODO Add parallelism?
        Value *Prev;
        for (size_t Bit = 0; Bit != BitWidth; ++Bit) {
          Value *BitCompare = Xor(GetBit(L, Bit), GetBit(R, Bit));
          Prev = Bit == 0 ? BitCompare : And(Prev, BitCompare);
        }
        NewBitMap(I).push_back(Prev);
        return;
      }
      case CmpInst::ICMP_UGT: // Same as next, fallthrough.
      case CmpInst::ICMP_SGT: visitAddSub(IsLeOpt, I, R, L); return;
      case CmpInst::ICMP_UGE: // Same as next, fallthrough.
      case CmpInst::ICMP_SGE: visitAddSub(IsLtOpt, I, R, L); return;
      case CmpInst::ICMP_ULT: // Same as next, fallthrough.
      case CmpInst::ICMP_SLT: visitAddSub(IsLtOpt, I, L, R); return;
      case CmpInst::ICMP_ULE: // Same as next, fallthrough.
      case CmpInst::ICMP_SLE: visitAddSub(IsLeOpt, I, L, R); return;
      default: {
        std::string Str;
        raw_string_ostream OS(Str);
        OS << "Unsupported compare predicate: " << I;
        report_fatal_error(OS.str());
      }
    }
  }
  void visitPHINode(PHINode &I) {
    dbgs() << "Visit " << I << "\n";
    size_t N = I.getNumIncomingValues();
    if (N != 2) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unimplemented PHI with more than 2 incomings: " << I;
      report_fatal_error(OS.str());
    }
    BasicBlock *LIn = I.getIncomingBlock(0), *RIn = I.getIncomingBlock(1);
    BasicBlock *LS = LIn->getSinglePredecessor();
    BasicBlock *RS = RIn->getSinglePredecessor();
    if (!LS || !RS) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unimplemented PHI non-single predecessor block: " << I;
      report_fatal_error(OS.str());
    }
    TerminatorInst *LT = LS->getTerminator(), *RT = RS->getTerminator();
    BranchInst *LB = dyn_cast<BranchInst>(LT), *RB = dyn_cast<BranchInst>(RT);
    if (!LB || !RB) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unimplemented predecessor not branch terminated: " << I;
      report_fatal_error(OS.str());
    }
    if (LB != RB) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unimplemented PHI with different predecessors: " << I;
      report_fatal_error(OS.str());
    }
    if (!LB->isConditional()) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << "Unimplemented PHI unconditional branch: " << I;
      report_fatal_error(OS.str());
    }
    BasicBlock *FalseBlock = LB->getSuccessor(0);
    BasicBlock *TrueBlock = LB->getSuccessor(1);
    Value *C = LB->getCondition();
    Value *L = I.getIncomingValue(0), *R = I.getIncomingValue(1);
    Value *T, *F;
    if (FalseBlock == LIn) {
      assert(TrueBlock == RIn && "Inconsistent incoming blocks");
      F = L;
      T = R;
    } else {
      assert(FalseBlock == RIn && "Inconsistent incoming blocks");
      assert(TrueBlock == LIn && "Inconsistent incoming blocks");
      F = R;
      T = L;
    }
    visitSelectPHI(C, T, F, I);
  }
  void visitTruncInst(TruncInst &I) {
    visitTruncCast(I);
  }
  void visitZExtInst(ZExtInst &I) {
    visitExt(false, I);
  }
  void visitSExtInst(SExtInst &I) {
    visitExt(true, I);
  }
  void visitBitCastInst(BitCastInst &I) {
    visitTruncCast(I);
  }
  void visitSelectInst(SelectInst &I) {
    dbgs() << "Visit " << I << "\n";
    Value *C = I.getCondition(), *T = I.getTrueValue(), *F = I.getFalseValue();
    visitSelectPHI(C, T, F, I);
  }

  // Fallthrough instruction visitor.
  void visitInstruction(Instruction &I) {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << "Unimplemented instruction: " << I;
    report_fatal_error(OS.str());
  }

};

// Needs to be a ModulePass to create new Functions and delete old ones.
struct BitBlast : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  BitBlast() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M) {
    std::vector<Function *> OldFunctions;
    std::vector<Function *> BooleanFunctions;

    for (Module::iterator F(M.begin()); F != M.end(); ++F) {

      if (std::find(BooleanFunctions.begin(), BooleanFunctions.end(), &*F) !=
          BooleanFunctions.end()) {
        dbgs() << "Skipping boolean function: " << F->getName() << "\n";
        continue;
      }

      dbgs() << "BitBlasting function " << F->getName() << ":" << *F << "\n";

      // Create the new function.
      BitBlastVisitor BBV(M, &*F);
      BBV.visit(F);
      Function *BoolF = BBV.F;
      BooleanFunctions.push_back(BoolF);

      dbgs() << "Created boolean function: " << BoolF->getName() << ":" <<
          *BoolF << "\n";

      OldFunctions.push_back(&*F);
    }

    // Delete the Module's non-boolean functions.
    assert(BooleanFunctions.size() == OldFunctions.size() &&
           "Should have created as many boolean functions as pre-existing "
           "functions in the Module");
    for (std::vector<Function *>::iterator F(OldFunctions.begin());
         F != OldFunctions.end(); ++F)
      (*F)->eraseFromParent();

    return BooleanFunctions.size() > 0;
  }
};

struct BitBlastPreCleanupVisitor :
      public InstVisitor<BitBlastPreCleanupVisitor> {
  bool Modified;

  BitBlastPreCleanupVisitor() : Modified(false) {}

  // TODO Handle URem with constant RHS that's ``(1 << s) - 1`` since
  //      inputs are of this form.
  void visitURem(BinaryOperator &I) {
    dbgs() << "Visiting: " << I << "\n";
    if (I.getType()->getPrimitiveSizeInBits() <= 32 &&
        expandRemainder(&I))
      return;
    // TODO Handle 64-bits. Note that a divisor of ``(1 << s) - 1`` is
    // easy to optimize. and our current sample input uses it.
    dbgs() << " --> Unhandled type\n";
  }
  void visitUDiv(BinaryOperator &I) {
    dbgs() << "Visiting: " << I << "\n";
    if (I.getType()->getPrimitiveSizeInBits() <= 32 &&
        expandDivisionUpTo32Bits(&I))
      return;
    // TODO Handle 64-bits.
    dbgs() << " --> Unhandled type\n";
  }
  void visitIRem(BinaryOperator &I) {
    dbgs() << "Visiting: " << I << "\n";
    if (I.getType()->getPrimitiveSizeInBits() <= 32 &&
        expandRemainder(&I))
      return;
    // TODO Handle 64-bits.
    dbgs() << " --> Unhandled type\n";
  }
  void visitIDiv(BinaryOperator &I) {
    dbgs() << "Visiting: " << I << "\n";
    if (I.getType()->getPrimitiveSizeInBits() <= 32 &&
        expandDivisionUpTo32Bits(&I))
      return;
    // TODO Handle 64-bits.
    dbgs() << " --> Unhandled type\n";
  }

  void visitMul(BinaryOperator &I) {
    dbgs() << "Visiting: " << I << "\n";
    Value *L = I.getOperand(0), *R = I.getOperand(1);
    if (!dyn_cast<ConstantInt>(R)) {
      if (dyn_cast<ConstantInt>(L))
        std::swap(L, R); // Later code assumes R is constant.
      else {
        // Neither side is constant, let BitBlast barf on it.
        dbgs() << " --> Bailing\n";
        return;
      }
    }
    Modified = true;
    IRBuilder<> B(I.getParent(), I);
    APInt C = cast<ConstantInt>(R)->getUniqueInteger();
    unsigned PowerOf2 = C.logBase2();
    Value *Replacement =
        B.CreateShl(L, ConstantInt::get(I.getType(), PowerOf2));
    for (APInt Remainder = C.urem(APInt(C.getBitWidth(), PowerOf2)); Remainder != 0; --Remainder)
      Replacement = B.CreateAdd(Replacement, L);
    I.replaceAllUsesWith(Replacement);
  }
  void visitLShr(BinaryOperator &I) {
    Value *L = I.getOperand(0), *R = I.getOperand(1);
    if (dyn_cast<ConstantInt>(R))
      // BitBlast can handle constant RHS.
      return;
    dbgs() << "Visiting: " << I << "\n";
    Modified = true;
    size_t BitWidth = TypeBitWidth(I);
    IRBuilder<> B(I.getParent(), I);
    // If UB where RHS > BitWidth, shift returns zero.
    Value *Previous = ConstantInt::get(I.getType(), 0);
    for (size_t Shift = 0; Shift != BitWidth; ++Shift) {
      Value *Shifted = Shift == 0 ? L : B.CreateLShr(L, Shift);
      Value *C = B.CreateICmpEQ(R, ConstantInt::get(I.getType(), Shift));
      Previous = B.CreateSelect(C, Shifted, Previous);
    }
    I.replaceAllUsesWith(Previous);
  }

};

struct BitBlastPreCleanup : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  BitBlastPreCleanup() : FunctionPass(ID) {}

  virtual bool runOnFunction(Function &F) {
      BitBlastPreCleanupVisitor BBPCUV;
      BBPCUV.visit(F);
      if (BBPCUV.Modified)
        dbgs() << F.getName() << " cleaned up: \n" << F << "\n";
      return BBPCUV.Modified;
  }
};

}

char BitBlast::ID = 0;
static RegisterPass<BitBlast>
Y1("BitBlast", "Transform functions to boolean formulas");

char BitBlastPreCleanup::ID = 0;
static RegisterPass<BitBlastPreCleanup>
Y2("PreBitBlast", "Cleanup to lower primitives BitBlast doesn't support");

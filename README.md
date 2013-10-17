BitBlast
========

This repository implements bitblasting of C/C++ code that can be proved
to terminated (i.e. loops have to be counted in some way): it transforms
arbitrary C/C++ functions into sequences of AND, OR and NOT
instructions.

It supports most arithmetic and logical operations as well as
conditionals on any integer bitwidth.

The repository also contains a tool which transforms such bitblasted
code into DIMACS Conjunctive Normal Form, which can then be fed to a SAT
solver to prove that different functions actually do implement the same
functionality for all possible inputs.

Implementation
--------------

The implementation relies heavily on LLVM's compiler infrastructure to
simplify some constructs before trying to bliblast them. Most
importantly, it relies on LLVM performing full loop unrolling because
bitblasted functions are only contain one giant basic block and don't
contain any control flow. It also importantly expects memory references
to be registerized, although the algorithm could be made to reason about
memory.

In a lesser way applying common subexpression elimination, instruction
combination, constant propagation, control flow graph simplifications
helps reduce boolean formula sizes. Scalar replacement of aggregates
allows the bitblasting code to only have to deal with bits as integer
values instead of having to know about structs and their layout.

An extra simplification pass occurs after this, to eliminate some
instructions that are cumbersome to handle while
bitblasting. Specifically division, modulus, multiplication of a
variable by a constant, shift right of a constant by a variable are all
simplified into easier-to-handle IR instructions.

At a low-level the bitblasting pass only emits AND, OR and NOT. XOR and
XNOR are offered alongside the other three primitives to allow for more
expressive code generation, but they're expressed in terms of the
previous three. These five primitives apply aggressive bit-level
constant propagation and memoization (effectively common subexpression
elimination) which the higher-level LLVM transforms don't reason about.

The higher-level LLVM IR instructions are all expressed in terms of
these five primitives: add, sub, and, or, xor, truncate, cast, bitcast,
zero-extend, sign-extend, select, PHI, left shift by constant,
arithmetic right shift by constant, logical right shift by constant,
return, branch, integer compare. The simplified LLVM functions are all
visited, and an equivalent function is generated using only primitive
operations.

More LLVM IR primitives could be supported (and some "by constant"
restrictions could be removed), but there hasn't been a need for this so
far.

Results
-------

The input.cpp file implements functions in a few different ways, and
also contains some CHECK functions which do pairwise comparisons of
these function's results. One way to test their equivalence would be to
call each CHECK function with all possible inputs, but the above
bitblasting and CNF tool allow us to instead feed a boolean formula to a
SAT solver and ask it to find inconsistencies.

Steps and results can be found in the opt and unopt directories. The
former applies constant propagation and memoization, whereas the later
doesn't.

The optimization pipeline manages to simplify much of the code and prove
some functions to be equivalent even before reaching the SAT
solver. Other functions are proven equivalent byt the SAT solver. One
function (the fourth parity implementation) is unsatisfied, indicating
that there is most likely a bug in the implementation of constant left
shift by a variable.

The functions which use modulo of 64-bit values are currently untested
because 64-bit modulo is currently unsupported. A call to LLVM's
runtime, which would then be inlined, should be a sufficient
implementation. Note that in all these cases a much simpler
simplification can be done because all 64-bit modulo use a constant
divisor which happens to be (1 << s) - 1, division is therefore not
necessary.

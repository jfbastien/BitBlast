#! /bin/bash

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <file>"
  exit 1
fi

#set -e
set -u
set -x

base=`basename "$1"`

clang "$1" -S -emit-llvm -o "$base.unopt"

# Run opt with options similar to -O3 but without some passes that
# aren't useful or that would hinder bitblasting (e.g. vectorization or
# loop idiom recognition). Also make sure that loop unrolling has no
# threshold (we want to remove all loops).
#
# To print -O3 passes:
#   llvm-as < /dev/null | opt -O3 -disable-output -debug-pass=Arguments
opt "$base.unopt" -S -o "$base.opt" \
  -no-aa \
  -tbaa \
  -targetlibinfo \
  -basicaa \
  -notti \
  -preverify \
  -domtree \
  -verify \
  -simplifycfg \
  -domtree \
  -sroa \
  -early-cse \
  -lower-expect \
  -targetlibinfo \
  -no-aa \
  -tbaa \
  -basicaa \
  -notti \
  -globalopt \
  -ipsccp \
  -deadargelim \
  -instcombine \
  -simplifycfg \
  -basiccg \
  -prune-eh \
  -inline-cost \
  -inline \
  -functionattrs \
  -argpromotion \
  -sroa \
  -domtree \
  -early-cse \
  -lazy-value-info \
  -jump-threading \
  -correlated-propagation \
  -simplifycfg \
  -instcombine \
  -tailcallelim \
  -simplifycfg \
  -reassociate \
  -domtree \
  -loops \
  -loop-simplify \
  -lcssa \
  -loop-rotate \
  -licm \
  -lcssa \
  -loop-unswitch \
  -instcombine \
  -scalar-evolution \
  -loop-simplify \
  -lcssa \
  -indvars \
  -loop-deletion \
  -unroll-threshold=0xffffffff \
  -loop-unroll \
  -memdep \
  -gvn \
  -memdep \
  -sccp \
  -instcombine \
  -lazy-value-info \
  -jump-threading \
  -correlated-propagation \
  -domtree \
  -memdep \
  -dse \
  -loops \
  -scalar-evolution \
  -adce \
  -simplifycfg \
  -instcombine \
  -domtree \
  -loops \
  -loop-simplify \
  -lcssa \
  -scalar-evolution \
  -loop-simplify \
  -lcssa \
  -instcombine \
  -simplifycfg \
  -strip-dead-prototypes \
  -globaldce \
  -constmerge \
  -preverify \
  -domtree \
  -verify

# Remove constructs BitBlast doesn't like.
opt "$base.opt" -S -o "$base.prebitblast" \
  -load=BitBlast.so -PreBitBlast

# Cleanup from the previous pass. Reduces code size to ~90% of original.
opt "$base.prebitblast" -S -o "$base.prebitblastclean" \
  -preverify -domtree -verify \
  -constprop -early-cse -gvn -adce \
  -preverify -domtree -verify

# The actual bit blasting.
# Add ``-BitBlast-memoize=0 -BitBlast-constprop=0`` to disable optimizations.
# The subsequent cleanup can't pick up all the optimizations done through
# memoization and constant propagation, leading to ~50% more code after cleanup.
opt "$base.prebitblastclean" -S -o "$base.bitblast" \
  -load=BitBlast.so -BitBlast

# Cleanup from the previous pass. Reduces code size to ~40% of original.
opt "$base.bitblast" -S -o "$base.bitblastclean" \
  -preverify -domtree -verify \
  -constprop -early-cse -gvn -adce \
  -preverify -domtree -verify

# Convert to CNF.
rm ./$base.cnf*
bc2cnf "$base.bitblastclean" -o "$base.cnf"
# Note: The above cleanup is too smart and most of the CHECK functions are true.
#       Use the following to ignore cleanup.
#bc2cnf "$base.bitblast" -o "$base.cnf"

# Run all CHECK functions through a SAT solver.
rm ./$base.cnf*CHECK*.sat
for file in $( ls ./$base.cnf*CHECK* ); do
  echo Processing $file
  cryptominisat $file >& $file.sat
  echo Result: $?
done

exit 0

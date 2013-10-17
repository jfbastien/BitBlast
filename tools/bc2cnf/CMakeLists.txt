set(LLVM_LINK_COMPONENTS support bitreader)

add_llvm_tool(bc2cnf
  bc2cnf.cpp
  )

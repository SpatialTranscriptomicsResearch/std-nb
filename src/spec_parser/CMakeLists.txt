find_package(FLEX)
find_package(BISON 3.0.4)

FLEX_TARGET(Lexer src/scanner.ll ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.cc)
BISON_TARGET(Parser src/parser.yy ${CMAKE_CURRENT_BINARY_DIR}/parser.tab.cpp)

ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

ADD_LIBRARY(parser SHARED
  src/Distribution.cpp
  src/Driver.cpp
  src/Formula.cpp
  src/RandomVariable.cpp
  src/RegressionEquation.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/parser.tab.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.cc
  )

TARGET_INCLUDE_DIRECTORIES(parser PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
TARGET_INCLUDE_DIRECTORIES(parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Needed for aux.hpp and coefficient.hpp, which don't get installed anywhere.
TARGET_INCLUDE_DIRECTORIES(parser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

INSTALL(TARGETS parser DESTINATION lib)

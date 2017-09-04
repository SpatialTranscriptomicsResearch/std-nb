find_package(FLEX)
find_package(BISON)

FLEX_TARGET(Lexer scanner.ll ${CMAKE_CURRENT_BINARY_DIR}/lex.yy.cc)
BISON_TARGET(Parser parser.yy ${CMAKE_CURRENT_BINARY_DIR}/parser.tab.cpp)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# Make it possible to #include generated header files for out-of-source builds
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_LIBRARY(transcriptomedeconvolution_parser STATIC
  driver.cpp
  lex.yy.cc
  parser.tab.cpp
  )

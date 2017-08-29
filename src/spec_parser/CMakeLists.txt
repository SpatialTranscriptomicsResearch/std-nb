ADD_CUSTOM_COMMAND(
  OUTPUT lex.yy.cc
  COMMAND flex ${CMAKE_CURRENT_SOURCE_DIR}/scanner.ll
  # rename output file in order for cmake to recognize it as a c++ file
  COMMAND cmake -E rename lex.yy.c lex.yy.cc
  )

ADD_CUSTOM_COMMAND(
  OUTPUT parser.tab.cc parser.tab.hh
  COMMAND bison -d ${CMAKE_CURRENT_SOURCE_DIR}/parser.yy
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

ADD_LIBRARY(transcriptomedeconvolution_parser STATIC
  driver.cpp
  lex.yy.cc
  parser.tab.cc
  )

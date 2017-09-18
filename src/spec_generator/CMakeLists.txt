ADD_EXECUTABLE(std-spec-generator src/spec_generator.cpp)
TARGET_LINK_LIBRARIES(std-spec-generator
  parser
  transcriptomedeconvolution
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  )
INSTALL(TARGETS std-spec-generator DESTINATION bin)

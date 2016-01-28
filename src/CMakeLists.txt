# add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -Wshadow -funroll-loops -Wno-unused-local-typedefs -march=native)

# Make it possible to #include generated header files for out-of-source builds
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_LIBRARY(multiScoop SHARED aux.cpp counts.cpp executioninformation.cpp
  io.cpp metropolis_hastings.cpp FactorAnalysis.cpp PoissonModel.cpp
  VariantModel.cpp sampling.cpp terminal.cpp timer.cpp pdist.cpp)

ADD_EXECUTABLE(multiScoop-bin main.cpp cli.cpp entropy.cpp)
ADD_EXECUTABLE(test_sampling test_sampling.cpp entropy.cpp)
SET_TARGET_PROPERTIES(multiScoop-bin PROPERTIES OUTPUT_NAME multiScoop)

TARGET_LINK_LIBRARIES(multiScoop-bin
  multiScoop
  m
  -L${Boost_LIBRARY_DIRS}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY})

TARGET_LINK_LIBRARIES(test_sampling
  multiScoop
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY})

IF(GOOGLEPERFTOOLS_FOUND)
  TARGET_LINK_LIBRARIES(multiScoop-bin ${PROFILER_LIBRARY} ${TCMALLOC_LIBRARY})
ENDIF()

CONFIGURE_FILE(git_config.hpp.in git_config.hpp)

INSTALL(TARGETS multiScoop-bin DESTINATION bin)
INSTALL(TARGETS multiScoop DESTINATION lib)

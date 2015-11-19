# add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -Wshadow -funroll-loops -Wno-unused-local-typedefs -march=native)

# Make it possible to #include <discrover_config.hpp> for out-of-source builds
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE(multiScoopIBP main.cpp io.cpp PoissonFactorAnalysis.cpp sampling.cpp montecarlo.cpp pdist.cpp)
ADD_EXECUTABLE(test_sampling test_sampling.cpp sampling.cpp)

TARGET_LINK_LIBRARIES(multiScoopIBP
  m
  -L${Boost_LIBRARY_DIRS}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY})


CONFIGURE_FILE(git_config.hpp.in git_config.hpp)

INSTALL(TARGETS multiScoopIBP DESTINATION bin)

# add definitions, compiler switches, etc.
ADD_DEFINITIONS(-Wall -Wshadow -funroll-loops -march=native)

# Make it possible to #include generated header files for out-of-source builds
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/spec_parser/include)

ADD_SUBDIRECTORY(spec_generator)
ADD_SUBDIRECTORY(spec_parser)

ADD_LIBRARY(std_jit SHARED
  jit.cpp
  log.cpp
  verbosity.cpp
)

TARGET_LINK_LIBRARIES(std_jit LLVM)

ADD_LIBRARY(transcriptomedeconvolution SHARED
  Experiment.cpp
  Model.cpp
  aux.cpp
  coefficient.cpp
  compression_mode.cpp
  counts.cpp
  covariate.cpp
  design.cpp
  entropy.cpp
  executioninformation.cpp
  gamma_func.cpp
  gp.cpp
  io.cpp
  metropolis_hastings.cpp
  modelspec.cpp
  optimization_method.cpp
  parameters.cpp
  pdist.cpp
  sampling.cpp
  terminal.cpp
  timer.cpp
  types.cpp
)

ADD_EXECUTABLE(std main.cpp cli.cpp)

SET(BUILD_TESTS false)
IF(BUILD_TESTS)
  ADD_EXECUTABLE(test_gamma_cdf test_gamma_cdf.cpp)
  ADD_EXECUTABLE(test_gamma_convolution test_gamma_convolution.cpp)
  ADD_EXECUTABLE(test_negative_binomial_convolution test_negative_binomial_convolution.cpp)
  # ADD_EXECUTABLE(test-log-beta test-log-beta.cpp pdist.cpp)
  # ADD_EXECUTABLE(test_sampling test_sampling.cpp)
ENDIF()

TARGET_LINK_LIBRARIES(transcriptomedeconvolution
  parser
  std_jit
  m
  -L${Boost_LIBRARY_DIRS}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_LOG_LIBRARY}
  ${Boost_THREAD_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_DATE_TIME_LIBRARY}
  )

TARGET_LINK_LIBRARIES(std transcriptomedeconvolution)
IF(BUILD_TESTS)
  TARGET_LINK_LIBRARIES(test_gamma_cdf transcriptomedeconvolution)
  TARGET_LINK_LIBRARIES(test_gamma_convolution transcriptomedeconvolution)
  TARGET_LINK_LIBRARIES(test_negative_binomial_convolution transcriptomedeconvolution)
  # TARGET_LINK_LIBRARIES(test-log-beta transcriptomedeconvolution)
  # TARGET_LINK_LIBRARIES(test_sampling transcriptomedeconvolution)
  # TARGET_LINK_LIBRARIES(test_field transcriptomedeconvolution)
ENDIF()

IF(GOOGLEPERFTOOLS_FOUND)
  TARGET_LINK_LIBRARIES(std ${PROFILER_LIBRARY} ${TCMALLOC_LIBRARY})
ENDIF()

CONFIGURE_FILE(git_config.hpp.in git_config.hpp)
SET_TARGET_PROPERTIES(std PROPERTIES RELEASE_POSTFIX ${CMAKE_RELEASE_POSTFIX})
SET_TARGET_PROPERTIES(std PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

INSTALL(TARGETS std DESTINATION bin)
INSTALL(TARGETS std_jit transcriptomedeconvolution DESTINATION lib)

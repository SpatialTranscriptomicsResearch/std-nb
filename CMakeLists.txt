

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  MESSAGE(FATAL_ERROR "Do not build in-source. "
    "Please remove CMakeCache.txt and the CMakeFiles/ directory. "
    "Then build out-of-source. "
    "For instructions, please refer to BUILDING.md.")
ENDIF()

PROJECT(multiScoopIBP)

# if you want the full compiler output, uncomment the following line
# SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")




# Change build configuration
#
# SET(CMAKE_BUILD_TYPE Debug)
# SET(CMAKE_BUILD_TYPE Release)

SET(CMAKE_RELEASE_POSTFIX -nxt)
SET(CMAKE_DEBUG_POSTFIX -nxt-dbg)

# Set installation destination and prefix path to find libraries installed on
# this sytem. If this is not specified, /usr/local/ will be used.
#
# SET(LOCAL_PREFIX "/usr/local")
# SET(CMAKE_INSTALL_PREFIX ${LOCAL_PREFIX})
# SET(CMAKE_PREFIX_PATH ${LOCAL_PREFIX})


SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_LINKER ld.gold)

INCLUDE(CheckCXXCompilerFlag)
INCLUDE(InstallRequiredSystemLibraries)

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Metropolis-Hastings MCMC for the multi-scoop Inidan Buffet Process aka. Beta-Negative Binomial Process")
SET(CPACK_PACKAGE_VENDOR "Jonas Maaskola")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "3")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
SET(CPACK_SOURCE_GENERATOR TGZ;TBZ2;ZIP)
SET(CPACK_GENERATOR TGZ;TBZ2;ZIP)
SET(CPACK_PACKAGE_CONTACT "Jonas Maaskola")
# SET(CPACK_STRIP_FILES "bin/MyExecutable")
# SET(CPACK_SOURCE_STRIP_FILES "")
# SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE(CPack)

SET(MULTISCOOPIBP_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

IF(EXISTS "${CMAKE_SOURCE_DIR}/local_configuration.cmake")
  INCLUDE("${CMAKE_SOURCE_DIR}/local_configuration.cmake")
ENDIF()

# Check whether the compiler supports position-independent code
CHECK_CXX_COMPILER_FLAG("-fPIC" COMPILER_SUPPORTS_PIC)
IF(COMPILER_SUPPORTS_PIC)
  ADD_DEFINITIONS("-fPIC")
ENDIF()
# TODO: for checking -fPIE it might be better to use something like
# CHECK_CXX_LINKER_FLAG; but such a macro does not exist
CHECK_CXX_COMPILER_FLAG("-fPIE -pie" COMPILER_SUPPORTS_PIE)
IF(COMPILER_SUPPORTS_PIE)
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIE -pie")
ENDIF()

# Check whether the compiler supports link-time optimization
IF(NOT(DEFINED WITH_LTO) OR WITH_LTO)
  CHECK_CXX_COMPILER_FLAG("-flto" COMPILER_SUPPORTS_LTO)
  IF(COMPILER_SUPPORTS_LTO)
    INCLUDE(CheckGCCVersion)
    IF(GCC_SUPPORTS_LTO)
      MESSAGE(STATUS "Enabled: link-time optimization")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    ELSE()
      IF(WITH_LTO)
        MESSAGE(FATAL_ERROR "Disabled: link-time optimization - LTO support in GCC version < 4.8 is unreliable")
      ELSE()
        MESSAGE(STATUS "Disabled: link-time optimization - LTO support in GCC version < 4.8 is unreliable")
      ENDIF()
    ENDIF()
  ELSE()
    IF(WITH_LTO)
      MESSAGE(FATAL_ERROR "Disabled: link-time optimization - no compiler support")
    ELSE()
      MESSAGE(STATUS "Disabled: link-time optimization - no compiler support")
    ENDIF()
  ENDIF()
ELSE()
  MESSAGE(STATUS "Disabled: link-time optimization (use -DWITH_LTO=ON to enable)")
ENDIF()



FIND_PACKAGE(LLVM REQUIRED CONFIG)

MESSAGE(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
MESSAGE(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
INCLUDE_DIRECTORIES(SYSTEM ${LLVM_INCLUDE_DIRS})
ADD_DEFINITIONS(${LLVM_DEFINITIONS})

# Boost configuration
# You may try uncommenting some of the commented options below if you have
# trouble automatically finding Boost on your system.

SET(BOOST_MIN_VERSION "1.58.0")
# Toggle verbose output of the boost configuration
#
SET(Boost_DEBUG OFF)

# If you have your Boost installation in a non-standard location, that is not
# part below ${LOCAL_PREFIX} and the automatic finding fails, you may set the
# Boost root directory here
#
# SET(BOOST_ROOT "~/local/boost-1.49.0")

# Uncomment if linking should not use system Boost installation but instead
# should prefer one in a non-standard location.
#
# SET(Boost_NO_SYSTEM_PATHS TRUE)

# SET(Boost_USE_STATIC_LIBS  OFF)
# SET(Boost_USE_STATIC_RUNTIME ON)
# SET(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0" "1.46" "1.46.0" "1.42.0" "1.40.0")
FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} COMPONENTS iostreams regex system log
  filesystem unit_test_framework program_options serialization date_time thread
  REQUIRED)
IF(Boost_FOUND)
  ADD_DEFINITIONS( "-DHAS_BOOST" )
  INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIR})
  LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
ENDIF()

FIND_PACKAGE(Eigen3 REQUIRED)
IF(EIGEN3_FOUND)
  MESSAGE(STATUS "Enabled: Eigen3 support")
  INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})
ELSE()
  MESSAGE(FATAL_ERROR "Eigen3 not found.")
ENDIF()

FIND_PACKAGE(LBFGS REQUIRED)
IF(LBFGS_FOUND)
  MESSAGE(STATUS "Enabled: LBFGS support")
  INCLUDE_DIRECTORIES(SYSTEM ${LBFGS_INCLUDE_DIR})
ELSE()
  MESSAGE(FATAL_ERROR "LBFGS.h not found.")
ENDIF()

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  MESSAGE(STATUS "Enabled: OpenMP support")
ELSE()
  MESSAGE(FATAL_ERROR "Compiler does not support OpenMP. See http://openmp.org/wp/openmp-compilers/ for a list of OpenMP supporting compilers.")
ENDIF()

IF(NOT(DEFINED WITH_TCMALLOC) OR WITH_TCMALLOC)
  FIND_PACKAGE(LibGooglePerfTools)
  IF(GOOGLEPERFTOOLS_FOUND)
    MESSAGE(STATUS "Enabled: usage of TCMALLOC")
  ELSE()
    IF(WITH_TCMALLOC)
      MESSAGE(FATAL_ERROR "TCMALLOC required but not found")
    ELSE()
      MESSAGE(STATUS "Disabled: usage of TCMALLOC - TCMALLOC not found")
    ENDIF()
  ENDIF()
ELSE()
  MESSAGE(STATUS "Disabled: usage of TCMALLOC (use -DWITH_TCMALLOC=ON to enable)")
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

INCLUDE(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
IF(GIT_SHA1 STREQUAL "GITDIR-NOTFOUND")
  MESSAGE(STATUS "This is not a build from a git clone.")
  SET(GIT_SHA1 "Not compiled from a git repository.")
  SET(GIT_DESCRIPTION
    "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
  SET(GIT_BRANCH "release")
ELSE()
  SET(GIT_BUILD 1)
  git_describe(GIT_DESCRIPTION --dirty)
  git_branch(GIT_BRANCH)
ENDIF()

ADD_SUBDIRECTORY(src)

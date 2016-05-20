cmake_minimum_required(VERSION 3.3)
project(Hubbard2DPomerol)
message("\n${PROJECT_NAME} configuration\n")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(TCLAP)
message(STATUS "TCLAP includes: " ${TCLAP_INCLUDE_DIR})
include_directories(${TCLAP_INCLUDE_DIR})

find_package(Boost 1.54.0 COMPONENTS mpi serialization REQUIRED)
message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}" )
message(STATUS "Boost libs: ${Boost_LIBRARIES}" )
include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS})

find_package (MPI)
message(STATUS "MPI includes: " ${MPI_CXX_INCLUDE_PATH} )
message(STATUS "MPI CXX libs: " ${MPI_CXX_LIBRARIES} )
message(STATUS "MPI definitions : " ${MPI_CXX_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS})
add_definitions(${MPI_CXX_COMPILE_FLAGS}  ${MPI_C_COMPILE_FLAGS})
include_directories(${PROJECT_NAME} PUBLIC ${MPI_CXX_INCLUDE_PATH})

find_package (Eigen3 3.1 REQUIRED)
message(STATUS "Eigen3 includes: " ${EIGEN3_INCLUDE_DIR} )
include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})

set(Pomerol_INCLUDE_DIR /usr/local/include)
include_directories(${PROJECT_NAME} PUBLIC ${Pomerol_INCLUDE_DIR})

set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -H")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPOMEROL_COMPLEX_MATRIX_ELEMENTS")

link_directories(/usr/local/lib)
set(SOURCE_FILES main.cpp)
add_executable(Hubbard2DPomerol ${SOURCE_FILES})

set(Pomerol_LIBRARIES /usr/local/lib/libpomerol.so)
message(STATUS "Pomerol_LIBRARIES: " ${Pomerol_LIBRARIES} )


target_link_libraries(Hubbard2DPomerol libpomerol.so ${Pomerol_LIBRARIES} ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES})

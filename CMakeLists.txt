if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(libdet)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


message(STATUS "OpenCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
include_directories(src)
include_directories(src/utils)
include_directories(src/runner/ax_type_header)

add_library(det SHARED
    src/libdet.cpp
    
    src/runner/axcl/axcl_manager.cpp
    src/runner/axcl/ax_model_runner_axcl.cpp
    src/runner/ax650/ax_model_runner_ax650.cpp
    src/utils/enum_devices.cpp
    src/ax_devices.cpp

    src/models/ax_model_base.cpp
)

target_link_libraries(det ${OpenCV_LIBS})

function(build_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} det)
endfunction()

build_test(test_ax_api)
build_test(test_axcl_api)
build_test(test_enum_devices)
build_test(test_det)

install(TARGETS det
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SYSTEM_PROCESSOR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY pydet/ DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)
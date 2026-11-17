file(GLOB_RECURSE PREEMPT_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_library(preempt SHARED ${PREEMPT_SRCS})
    set_target_properties(preempt PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    install(TARGETS preempt EXPORT XSchedTargets LIBRARY DESTINATION lib)
    install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_library(preempt STATIC ${PREEMPT_SRCS})
    set_target_properties(preempt PROPERTIES
        SUFFIX ".dll"           # Ensure that the dynamic library suffix is .dll
        IMPORT_SUFFIX ".lib"    # Mandatory import of library suffix .lib
    )
endif()

add_library(XSched::preempt ALIAS preempt)

target_include_directories(preempt PUBLIC
    $<BUILD_INTERFACE:${XSCHED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(preempt PRIVATE
    -fPIC
    -Wall
    -Wextra
    -Werror
)

target_link_libraries(preempt
    PRIVATE utils
    PRIVATE sched
    PRIVATE protocol
)

if(BUILD_TEST)
    add_executable(async_xqueue_control_test test/async_xqueue_control.cpp)
    target_link_libraries(async_xqueue_control_test PRIVATE preempt protocol utils)
    target_compile_options(async_xqueue_control_test PRIVATE -Wall -Wextra -Werror)
    add_test(NAME async_xqueue_control_test COMMAND async_xqueue_control_test)
    set_tests_properties(async_xqueue_control_test PROPERTIES
        ENVIRONMENT "XSCHED_SCHEDULER=APP" TIMEOUT 30)
endif()

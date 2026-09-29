include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
    server-warm-tier.cpp
    server-warm-tier.h
    server-warm-tier-executors.cpp
    server-warm-tier-executors.h
    server-warm-tier-runtime.cpp
    server-warm-tier-runtime.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

option(S41_SERVER_FFN_SPLIT "Build the experimental server FFN split client" OFF)
if(S41_SERVER_FFN_SPLIT)
    if(NOT UNIX OR ANDROID)
        message(FATAL_ERROR "S41_SERVER_FFN_SPLIT requires host libusb")
    endif()
    find_library(S41_SERVER_FFN_LIBUSB NAMES usb-1.0 libusb-1.0.so.0 REQUIRED)
    target_sources(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
        ${CMAKE_SOURCE_DIR}/examples/layersplit/ffn-split-client.cpp
        ${CMAKE_SOURCE_DIR}/examples/layersplit/ffn-split-usb-client.cpp
    )
    target_compile_definitions(${TARGET} PRIVATE
        S41_SERVER_FFN_SPLIT
        FFN_SPLIT_USB_TRANSPORT
    )
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/examples/gguf-hash/deps)
    target_link_libraries(${TARGET} PRIVATE ${S41_SERVER_FFN_LIBUSB})
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(BUILD_TESTING)
    add_executable(test-warm-tier-executors
        tests/test-warm-tier-executors.cpp
    )
    target_link_libraries(test-warm-tier-executors PRIVATE llama-server-impl)
    add_test(NAME test-warm-tier-executors COMMAND test-warm-tier-executors)
endif()

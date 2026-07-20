if (NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR ANDROID))
    message(STATUS "Skipping experimental phone-pim examples outside Linux/Android")
    return()
endif()

add_library(phone-pim-protocol STATIC
    phone_pim_protocol.cpp
    phone_pim_socket.cpp
)
target_include_directories(phone-pim-protocol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash/deps/sha256
)
target_link_libraries(phone-pim-protocol PRIVATE sha256 ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(phone-pim-protocol PRIVATE cxx_std_17)

add_library(phone-pim-client STATIC phone_pim_client.cpp)
target_link_libraries(phone-pim-client PRIVATE phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(phone-pim-client PRIVATE cxx_std_17)

add_library(phone-pim-store STATIC phone_pim_store.cpp)
target_include_directories(phone-pim-store PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash/deps/sha256
)
target_link_libraries(phone-pim-store PRIVATE phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(phone-pim-store PRIVATE cxx_std_17)

add_library(phone-pim-island STATIC phone_pim_ffn.cpp)
target_include_directories(phone-pim-island PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash/deps/sha256
)
target_link_libraries(phone-pim-island PRIVATE ggml sha256 ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(phone-pim-island PRIVATE cxx_std_17)

set(TARGET llama-phone-pim-worker)
add_executable(${TARGET} phone_pim_worker.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE phone-pim-protocol phone-pim-store phone-pim-island ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-phone-pim-host)
add_executable(${TARGET} phone_pim_host.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash/deps/sha256
)
target_link_libraries(${TARGET} PRIVATE phone-pim-protocol phone-pim-store phone-pim-island llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-phone-pim-fleet)
add_executable(${TARGET} phone_pim_fleet.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE phone-pim-client phone-pim-protocol phone-pim-island ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-phone-pim-bench)
add_executable(${TARGET} phone_pim_bench.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash/deps/sha256
)
target_link_libraries(${TARGET} PRIVATE phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (BUILD_TESTING)
    set(TARGET test-phone-pim-protocol)
    add_executable(${TARGET} test_phone_pim_protocol.cpp)
    target_link_libraries(${TARGET} PRIVATE phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME phone-pim-protocol COMMAND ${TARGET})

    set(TARGET test-phone-pim-client)
    add_executable(${TARGET} test_phone_pim_client.cpp)
    target_link_libraries(${TARGET} PRIVATE phone-pim-client phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME phone-pim-client COMMAND ${TARGET})

    add_test(NAME phone-pim-fleet-cli-empty COMMAND llama-phone-pim-fleet)
    set_tests_properties(phone-pim-fleet-cli-empty PROPERTIES WILL_FAIL TRUE)
    add_test(
        NAME phone-pim-fleet-cli-duplicate-physical
        COMMAND ${CMAKE_COMMAND}
            -DFLEET=$<TARGET_FILE:llama-phone-pim-fleet>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test_phone_pim_fleet_cli.cmake
    )

    set(TARGET test-phone-pim-store)
    add_executable(${TARGET} test_phone_pim_store.cpp)
    target_link_libraries(${TARGET} PRIVATE phone-pim-store phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    add_test(NAME phone-pim-store COMMAND ${TARGET})

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(TARGET test-phone-pim-stream)
        add_executable(${TARGET} test_phone_pim_stream.cpp)
        target_link_libraries(${TARGET} PRIVATE phone-pim-store phone-pim-protocol ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(${TARGET} PRIVATE cxx_std_17)
        add_test(
            NAME phone-pim-stream
            COMMAND ${TARGET} $<TARGET_FILE:llama-phone-pim-worker>
        )
        set_tests_properties(phone-pim-stream PROPERTIES TIMEOUT 30)
    endif()
endif()

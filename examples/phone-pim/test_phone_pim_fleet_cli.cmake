if (NOT DEFINED FLEET)
    message(FATAL_ERROR "FLEET is required")
endif()

execute_process(
    COMMAND "${FLEET}"
        --device a,127.0.0.1,1,same
        --device b,127.0.0.1,2,same
        --model missing.gguf --jobs 2
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if (result EQUAL 0)
    message(FATAL_ERROR "duplicate physical device ID was accepted")
endif()
if (NOT error MATCHES "error: duplicate physical device ID")
    message(FATAL_ERROR "wrong rejection path: ${error}")
endif()
if (error MATCHES "local model identity")
    message(FATAL_ERROR "model I/O occurred before duplicate-device rejection")
endif()

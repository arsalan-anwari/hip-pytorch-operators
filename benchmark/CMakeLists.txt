function(add_benchmark BENCHMARK_SOURCE)
  get_filename_component(BENCHMARK_TARGET ${BENCHMARK_SOURCE} NAME_WE)

  if(BENCHMARK_CONFIG_TUNING)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BENCHMARK_TARGET}.parallel.cpp.in")
      message(STATUS "found ${BENCHMARK_TARGET}.parallel.cpp.in file, compiling in parallel.")
      read_config_autotune_settings(${BENCHMARK_TARGET} list_across_names list_across output_pattern_suffix)
      #make sure that variables are not empty, i.e. there actually is an entry for that benchmark in benchmark/ConfigAutotuneSettings.cmake
      if(list_across_names)
        add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
        target_compile_definitions(${BENCHMARK_TARGET} PRIVATE BENCHMARK_CONFIG_TUNING)
        add_matrix(TARGET ${BENCHMARK_TARGET}
                SHARDS 1
                CURRENT_SHARD 0
                INPUT "${BENCHMARK_TARGET}.parallel.cpp.in"
                OUTPUT_PATTERN "${BENCHMARK_TARGET}_${output_pattern_suffix}"
                NAMES ${list_across_names}
                LISTS ${list_across})
        add_dependencies(benchmark_config_tuning ${BENCHMARK_TARGET})
      else()
        message(WARNING "No config-tuning entry in benchmark/ConfigAutotuneSettings.cmake for ${BENCHMARK_TARGET}!")
        return()
      endif()
    else()
      #do nothing if BENCHMARK_CONFIG_TUNING is ON but no ${BENCHMARK_TARGET}.parallel.cpp.in exists
      return()
    endif()
  else()
    add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
  endif()

  if(BUILD_NAIVE_BENCHMARK)
    target_compile_definitions(${BENCHMARK_TARGET} PUBLIC BUILD_NAIVE_BENCHMARK)
  endif()

  target_link_libraries(${BENCHMARK_TARGET}
    PRIVATE
      hip-pytorch-operators
      benchmark::benchmark
  )
  if(NOT USE_HIP_CPU)
    target_link_libraries(${BENCHMARK_TARGET}
      PRIVATE
        hip-pytorch-operators_hip
    )
  else()
    target_link_libraries(${BENCHMARK_TARGET}
      PRIVATE
        Threads::Threads
        hip_cpu_rt::hip_cpu_rt
    )
    if(STL_DEPENDS_ON_TBB)
      target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
          TBB::tbb
      )
    endif()
  endif()

  target_compile_options(${BENCHMARK_TARGET}
    PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:
        /bigobj # number of sections exceeded object file format limit: compile with /bigobj
      >
  )

  set_target_properties(${BENCHMARK_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
  )

  if (WIN32 AND NOT DEFINED DLLS_COPIED)
    set(DLLS_COPIED "YES")
    set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
    # for now adding in all .dll as dependency chain is not cmake based on win32
    file( GLOB third_party_dlls
    LIST_DIRECTORIES ON
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach( file_i ${third_party_dlls})
      add_custom_command( TARGET ${BENCHMARK_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/benchmark )
    endforeach( file_i )
  endif()
endfunction()

# ****************************************************************************
# Benchmarks
# ****************************************************************************

add_benchmark(benchmark_torch_triu.cpp)

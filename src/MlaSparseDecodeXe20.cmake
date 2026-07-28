# Generate Sparse MLA decode kernel instantiation files for DeepSeek V4.
# Each ELEM_TAG is compiled as a separate library to parallelize compilation.

set(MLA_SPARSE_DECODE_ELEM_TAGS half bf16)
set(MLA_SPARSE_DECODE_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")

set(MLA_SPARSE_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_sparse_decode_kernel.cpp.in")

list(LENGTH MLA_SPARSE_DECODE_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

set(MLA_SPARSE_DECODE_2STAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_sparse_decode_2stage_kernel.cpp.in")

# The 2-stage template generates one TU per (ELEM_TAG, D_QK, B_H, HAS_ATTN_SINK),
# mirroring the fused MLA decode path's per-(ELEM_TAG, PAGE_SIZE) split above. D_QK is
# the QK head dim (always 512 for decode) and B_H the sparse-decode analog of page
# size: together they key the Stage-2 config; HAS_ATTN_SINK selects the sink epilogue
# variant. One variant per object file bounds per-file compilation memory (avoids the
# build OOM guard -- one sink variant per file instead of both). The op dispatches
# dtype, then D_QK, then B_H, then the runtime attn_sink flag.
set(MLA_SPARSE_DECODE_2STAGE_D_QK 512)
set(MLA_SPARSE_DECODE_2STAGE_B_H 8 16 32 64)
set(MLA_SPARSE_DECODE_2STAGE_HAS_ATTN_SINK 0 1)

foreach(_idx RANGE ${_num_elems})
    list(GET MLA_SPARSE_DECODE_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET MLA_SPARSE_DECODE_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    # Fused (single-pass) variant -- optimization track, opt-in via USE_MLA_SPARSE_FUSED
    # (default OFF). The 2-stage variant below is always built (shipping default).
    if(USE_MLA_SPARSE_FUSED)
        set(GENERATED_FILE
            "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_sparse_decode_kernel_${ELEM_TAG}_128.cpp")
        configure_file(${MLA_SPARSE_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
        list(APPEND device_cpp_xe20 ${GENERATED_FILE})
    endif()

    # Two-stage: one TU per (ELEM_TAG, D_QK, B_H, HAS_ATTN_SINK).
    foreach(D_QK ${MLA_SPARSE_DECODE_2STAGE_D_QK})
        foreach(B_H ${MLA_SPARSE_DECODE_2STAGE_B_H})
            foreach(HAS_ATTN_SINK ${MLA_SPARSE_DECODE_2STAGE_HAS_ATTN_SINK})
                set(GENERATED_FILE_2STAGE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_sparse_decode_2stage_kernel_${ELEM_TAG}_${D_QK}_${B_H}_${HAS_ATTN_SINK}.cpp")
                configure_file(${MLA_SPARSE_DECODE_2STAGE_TEMPLATE} ${GENERATED_FILE_2STAGE} @ONLY)
                list(APPEND device_cpp_xe20 ${GENERATED_FILE_2STAGE})
            endforeach()
        endforeach()
    endforeach()
endforeach()

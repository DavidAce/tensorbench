add_library(tb-tid-o OBJECT)
target_sources(tb-tid-o PRIVATE tid.cpp ur.cpp token.cpp)
target_link_libraries(tb-tid-o PUBLIC tb-deps tb-flags)
if (TB_ENABLE_PCH)
    target_precompile_headers(tb-tid-o REUSE_FROM tb-pch)
endif ()

#add_library(tb-tid INTERFACE)
#target_link_libraries(tb-tid INTERFACE $<TARGET_OBJECTS:tb-tid-o>)
#target_link_libraries(tb-tid INTERFACE tb-deps tb-flags)


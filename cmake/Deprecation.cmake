
if(TB_DOWNLOAD_METHOD)
    message(FATAL_ERROR "The variable [TB_DOWNLOAD_METHOD] has been deprecated. Replace by:\n"
            "TB_PACKAGE_MANAGER:STRING=[find|cmake|fetch|find-or-cmake|find-or-fetch|conan]")
endif()

if(TB_DEPS_IN_SUBDIR)
    message(DEPRECATION "The option [TB_DEPS_IN_SUBDIR] will be ignored.")
endif()

if(TB_PRINT_INFO)
    message(DEPRECATION "The option [TB_PRINT_INFO] has been deprecated. Replace by:\n"
            "Use the built-in CMake CLI option --loglevel=[TRACE|DEBUG|VERBOSE|STATUS...] instead")
endif()

if(TB_PROFILE_BUILD)
    message(FATAL_ERROR "The option [TB_PROFILE_BUILD] has been deprecated. Replace by:\n"
            "COMPILER_PROFILE_BUILD:BOOL=[TRUE|FALSE]")
endif()


if(TB_ENABLE_ACRO)
    message(FATAL_ERROR "The option [TB_ENABLE_ACRO] has been deprecated.")
endif()

if(TB_ENABLE_MKL)
    message(FATAL_ERROR "The option [TB_ENABLE_MKL] has been deprecated.")
endif()

if(TB_ENABLE_OPENBLAS)
    message(FATAL_ERROR "The option [TB_ENABLE_OPENBLAS] has been deprecated.")
endif()

if(TB_PACKAGE_MANAGER)
    message(FATAL_ERROR "The option [TB_PACKAGE_MANAGER] has been deprecated.")
endif()

if(TB_ENABLE_ASAN)
    message(FATAL_ERROR "The option [TB_ENABLE_ASAN] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_ASAN:BOOL=[TRUE|FALSE]")
endif()

if(TB_ENABLE_USAN)
    message(FATAL_ERROR "The option [TB_ENABLE_USAN] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_USAN:BOOL=[TRUE|FALSE]")
endif()

if(TB_ENABLE_LTO)
    message(FATAL_ERROR "The option [TB_ENABLE_LTO] has been deprecated. Replace by:\n"
            "CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=[TRUE|FALSE]")
endif()

if(TB_ENABLE_PCH)
    message(FATAL_ERROR "The option [TB_ENABLE_PCH] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_PCH:BOOL=[TRUE|FALSE]")
endif()

if(TB_ENABLE_CCACHE)
    message(FATAL_ERROR "The option [TB_ENABLE_CCACHE] has been deprecated. Replace by:\n"
            "COMPILER_ENABLE_CCACHE:BOOL=[TRUE|FALSE]")
endif()

if(COMPILER_MARCH)
    message(FATAL_ERROR "The option [COMPILER_MARCH] has been deprecated.")
endif()
if(COMPILER_MTUNE)
    message(FATAL_ERROR "The option [COMPILER_MTUNE] has been deprecated.")
endif()

if(COMPILER_ENABLE_LTO)
    message(FATAL_ERROR "The option [COMPILER_ENABLE_LTO] has been deprecated.. Replace by:\n"
            "CMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=[TRUE|FALSE]")
endif()
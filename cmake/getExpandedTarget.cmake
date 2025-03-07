function(get_imported_location location target_name)
    get_target_property(PROP_IMP ${target_name} IMPORTED)
    get_target_property(PROP_TYP ${target_name} TYPE)
    if(PROP_IMP AND NOT PROP_TYP MATCHES "INTERFACE")
        string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
        set(PROP_CFGS _${BUILD_TYPE};_${CMAKE_BUILD_TYPE};)
#        set(PROP_VARS IMPORTED_LOCATION;LOCATION)
        set(PROP_VARS IMPORTED_LOCATION)
        foreach(PROP_CFG ${PROP_CFGS})
            foreach(PROP_VAR ${PROP_VARS})
                set(PROP ${PROP_VAR}${PROP_CFG})
                message(STATUS "target_name : ${target_name}")
                message(STATUS "PROP        : ${PROP}")
                message(STATUS "PROP_VAR    : ${PROP_VAR}")
                message(STATUS "PROP_CFG    : ${PROP_CFG}")
                get_property(PROP_DEF TARGET "${target_name}" PROPERTY "${PROP}" DEFINED)
                message(STATUS "PROP_DEF    : ${PROP_DEF}")

                get_property(PROP_SET TARGET "${target_name}" PROPERTY "${PROP}" SET)
                message(STATUS "PROP_SET    : ${PROP_SET}")
                if(PROP_SET)
                    get_target_property(PROP_LOC ${target_name} "${PROP}")
                    if(${PROP_LOC} MATCHES "NOTFOUND")
                        unset(PROP_LOC)
                    else()
                        set(${location} "${PROP_LOC}" PARENT_SCOPE)
                        return()
                    endif()
                endif()
            endforeach()
        endforeach()
    endif()
endfunction()


function(expand_target_all_targets target_names expanded_list)
    foreach(target_name ${target_names})
        if(TARGET ${target_name} AND NOT ${target_name} IN_LIST expanded_list)
            list(APPEND target_names_expanded ${target_name})
            unset(interface_libs)
            unset(private_libs)
            unset(imported_lib)
            unset(lib_type)
            get_target_property(lib_type ${target_name} TYPE)
            get_imported_location(imported_lib ${target_name})
            if(NOT lib_type MATCHES "INTERFACE")
                get_target_property(private_libs ${target_name} LINK_LIBRARIES)
            endif()
            get_target_property(interface_libs ${target_name} INTERFACE_LINK_LIBRARIES)
            list(FILTER imported_lib EXCLUDE REGEX "NOTFOUND|(-o)$")
            list(FILTER private_libs EXCLUDE REGEX "NOTFOUND|(-o)$")
            list(FILTER interface_libs EXCLUDE REGEX "NOTFOUND|(-o)$")
            foreach(elem ${imported_lib};${private_libs};${interface_libs})
                string(REGEX REPLACE "([\$<]+[A-Za-z]+:[A-Za-z]+[:>]+)|:>|>" "" elem_stripped "${elem}")
                if(NOT TARGET ${elem_stripped})
                    continue()
                endif()
                if(${elem_stripped} IN_LIST target_names)
                    continue()
                endif()
                if(${elem} IN_LIST expanded_list)
                    continue()
                endif()
                if(${elem_stripped} IN_LIST target_names_expanded)
                    continue()
                endif()
                unset(recursed_list) # Otherwise this one grows for each elem
                expand_target_all_targets(${elem_stripped} recursed_list)
                list(REMOVE_DUPLICATES "recursed_list")
                foreach(rec ${recursed_list})
                    if(NOT TARGET ${rec})
                        continue()
                    endif()
                    if(${rec} IN_LIST target_names)
                        continue()
                    endif()
                    if(${rec} IN_LIST expanded_list)
                        continue()
                    endif()
                    if(${rec} IN_LIST target_names_expanded)
                        continue()
                    endif()
                    list(APPEND target_names_expanded ${rec})
                endforeach()
            endforeach()
        endif()
    endforeach()

    # Remove duplicates in a way that retains linking order, i.e. keep last occurrence
    if(target_names_expanded)
        list(REVERSE target_names_expanded)
        list(REMOVE_DUPLICATES "target_names_expanded")
        list(REVERSE "target_names_expanded")
        list(APPEND ${target_names_expanded} "${${expanded_list}}")
    endif()
    list(APPEND ${expanded_list} "${target_names_expanded}")
    set(${expanded_list} "${${expanded_list}}" PARENT_SCOPE)
endfunction()

#pragma once

namespace debug {
    inline int exit_status = 0;
    void       register_callbacks();
    void       signal_callback_handler(int signum);
    void       print_stack_trace();
}
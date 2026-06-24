#include "backend.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <thread>

namespace free_splatter {

namespace {

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// Default CPU threads = physical cores. SMT siblings only add contention for
// matmul-heavy work, so on x86 (logical == 2x physical) we halve -- but ONLY
// when SMT is actually on. ARM / Apple silicon have no SMT, where a blanket /2
// silently throws away half the cores.
unsigned default_cpu_threads() {
    unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    std::ifstream smt("/sys/devices/system/cpu/smt/active");
    int on = 0;
    if (smt >> on && on == 1) {
        return std::max(1u, logical / 2);
    }
    return logical;
}

// "cuda:1" -> ("cuda", 1); "vulkan" -> ("vulkan", 0); "" -> ("", 0).
void parse_device(const std::string & req, std::string & name, int & index) {
    const size_t colon = req.find(':');
    name  = lower(colon == std::string::npos ? req : req.substr(0, colon));
    index = (colon != std::string::npos) ? std::atoi(req.c_str() + colon + 1) : 0;
}

// Discover dynamically-loadable backends once. For a GGML_BACKEND_DL +
// GGML_CPU_ALL_VARIANTS build this loads every libggml-cpu-<isa>.so and ggml
// scores them, so the host's best ISA is selected at run time. No-op for a
// statically-linked build, where backends register at static-init.
void load_backends_once() {
    static const bool done = [] { ggml_backend_load_all(); return true; }();
    (void) done;
}

// Set threads through the backend registry rather than ggml_backend_cpu_set_n_threads:
// in a DL build that symbol lives in the variant .so, not in the linked base.
void set_cpu_threads(ggml_backend_t be, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
    auto set_fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_fn) set_fn(be, n_threads);
}

} // namespace

bool engine_backend::init(const std::string & device_req, int n_threads) {
    release();
    load_backends_once();

    std::string name;
    int         want_idx = 0;
    parse_device(device_req, name, want_idx);

    if (name.empty() || name == "cpu") {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!be) {
            error = "CPU backend init failed";
            return false;
        }
        device = "cpu";
        if (const char * env = std::getenv("FREE_SPLATTER_NTHREADS")) {
            if (int v = std::atoi(env)) n_threads = v;  // explicit override; 0 = auto
        }
        if (n_threads <= 0) {
            n_threads = (int) default_cpu_threads();
        }
        set_cpu_threads(be, n_threads);
    } else if (name == "gpu" || name == "cuda" || name == "vulkan") {
        // "gpu" picks the first GPU of whichever backend was compiled in;
        // "cuda"/"vulkan" pin a specific backend when more than one is built.
        // ":N" selects the Nth matching GPU.
        const std::string want_reg = (name == "gpu") ? "" : name;
        int gpu_idx = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (!want_reg.empty()) {
                const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
                if (!reg || lower(reg) != want_reg) continue;
            }
            if (gpu_idx++ != want_idx) continue;
            be = ggml_backend_dev_init(dev, nullptr);
            if (be) {
                device = ggml_backend_dev_name(dev);
                break;
            }
        }
        if (!be) {
            error = "no usable '" + name +
                    "' device (built with FREE_SPLATTER_CUDA/FREE_SPLATTER_VULKAN? driver present?)";
            return false;
        }
    } else {
        error = "unknown device '" + device_req + "' (want cpu|gpu|cuda|vulkan)";
        return false;
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "gallocr init failed";
        release();
        return false;
    }
    return true;
}

void engine_backend::release() {
    if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
    if (be)     { ggml_backend_free(be);     be = nullptr; }
    device.clear();
}

bool engine_backend::is_cpu() const {
    return be && ggml_backend_dev_type(ggml_backend_get_device(be)) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

} // namespace free_splatter

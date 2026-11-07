// The device-side floor of a kernel on the A770 (README section 5): the profiled duration of an empty kernel and of small elementwise kernels, by
// launch size, and the host time per launch back to back. Tells whether the ~11 microseconds a
// 64x64 matrix product takes on the card is the cost of any kernel or of the product's kernel.
//
//   source /opt/intel/oneapi/setvars.sh   # in a shell of its own: it breaks PyTorch XPU imports
//   icpx -fsycl -O3 kernel_floor.cpp -o kernel_floor && ./kernel_floor
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>

using clk = std::chrono::steady_clock;

static double event_us(const sycl::event& e) {
  return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.0;
}

int main() {
  sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
  float* x = sycl::malloc_device<float>(1 << 24, q);
  q.fill(x, 1.0f, 1 << 24).wait();
  struct Case { const char* name; size_t global, local; bool work; };
  for (Case c : {Case{"empty, 1 item", 1, 1, false}, Case{"empty, 256 items", 256, 256, false},
                 Case{"empty, 4096 groups of 256", 1 << 20, 256, false}, Case{"add, 4096 elements", 4096, 256, true},
                 Case{"add, 1M elements", 1 << 20, 256, true}, Case{"add, 16M elements", 1 << 24, 256, true}}) {
    auto launch = [&] {
      return q.parallel_for(sycl::nd_range<1>(c.global, c.local), [=, w = c.work](sycl::nd_item<1> it) {
        if (w) x[it.get_global_id(0)] += 1.0f;
      });
    };
    for (int i = 0; i < 20; ++i) launch().wait();
    auto t0 = clk::now();
    sycl::event last;
    for (int i = 0; i < 5000; ++i) last = launch();
    last.wait();
    double host = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / 5000;
    double dev = 0;
    for (int i = 0; i < 50; ++i) { auto e = launch(); e.wait(); dev += event_us(e); }
    std::printf("{\"case\": \"%s\", \"host_us_per_launch\": %.2f, \"kernel_us\": %.2f}\n", c.name, host, dev / 50);
  }
}

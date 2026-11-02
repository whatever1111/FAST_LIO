// The negative control of the front-end ThreadSanitizer phase (CMakeLists.txt, FASTLIO_TEST_SANITIZER).
//
// Two threads write one int with nothing ordering the writes: a data race by construction. The program is
// built only with FASTLIO_TEST_SANITIZER=thread, through the same target-level function that instruments
// the node test component, and is never registered with CTest. The parent repository's
// `tools/ci/run_local.sh sanitize tsan` runs it under the phase's own preload, suppressions and
// `setarch -R`, and requires "WARNING: ThreadSanitizer: data race" and a nonzero exit before it trusts a
// clean run of the front-end tests: a build in which the instrumentation did not take effect fails there.
//
// The writes are apart in time but not ordered for TSan. The writer raises a relaxed flag after its
// increment, and the main thread waits for the flag before its own. A relaxed atomic gives TSan no
// happens-before edge, so the race remains, and every run reports it. Without the wait the two increments
// can run at the same moment on two cores, and TSan, whose shadow memory is updated without locks, can
// miss a race between simultaneous accesses.
#include <atomic>
#include <cstdio>
#include <thread>

namespace
{
int racedCounter = 0;                  // written by both threads below without synchronization, on purpose
std::atomic<bool> writerDone{false};  // relaxed only: it orders the writes in time, not for TSan
}

int main()
{
  std::thread writer([] {
    ++racedCounter;
    writerDone.store(true, std::memory_order_relaxed);
  });
  while (!writerDone.load(std::memory_order_relaxed)) {
    std::this_thread::yield();
  }
  ++racedCounter;  // after the writer's increment in time, but unordered with it: a data race
  writer.join();
  std::printf("tsan negative control: counter=%d\n", racedCounter);
  return 0;
}

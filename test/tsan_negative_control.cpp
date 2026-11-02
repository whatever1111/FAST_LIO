// The negative control of the front-end ThreadSanitizer phase (CMakeLists.txt, FASTLIO_TEST_SANITIZER).
//
// Two threads write one int with nothing ordering the writes: a data race by construction. The program is
// built only with FASTLIO_TEST_SANITIZER=thread, through the same target-level function that instruments
// the node test component, and is never registered with CTest. The parent repository's
// `tools/ci/run_local.sh sanitize tsan` runs it under the phase's own preload, suppressions and
// `setarch -R`, and requires "WARNING: ThreadSanitizer: data race" and a nonzero exit before it trusts a
// clean run of the front-end tests: a build in which the instrumentation did not take effect fails there.
#include <cstdio>
#include <thread>

namespace
{
int racedCounter = 0;  // written by both threads below without synchronization, on purpose
}

int main()
{
  std::thread writer([] { ++racedCounter; });
  ++racedCounter;  // after the thread starts, before the join: unordered with the writer's increment
  writer.join();
  std::printf("tsan negative control: counter=%d\n", racedCounter);
  return 0;
}

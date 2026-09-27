#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>

#include "frontend_observation.hpp"

namespace
{
fast_lio::FrontendFacts healthy()
{
  fast_lio::FrontendFacts facts;
  facts.completed = true;
  facts.initialized = true;
  facts.contextKnown = true;
  facts.stampKnown = true;
  facts.pose[6] = 1.0;
  return facts;
}

TEST(FrontendObservation, PublicationFaultsAreStickyAndNeverEscape)
{
  for (const bool unknown : {false, true}) {
    fast_lio::FrontendPublicationGuard guard;
    int payloadCalls = 0;
    int reportCalls = 0;
    const auto fail = [&] {
      ++payloadCalls;
      if (unknown)
        throw 7;
      throw std::runtime_error("injected publication failure");
    };
    const auto report = [&](const char * reason) {
      ++reportCalls;
      EXPECT_NE(reason, nullptr);
      throw std::runtime_error("logger failure must also remain isolated");
    };
    EXPECT_FALSE(guard.run(fail, report));
    EXPECT_TRUE(guard.faulted());
    EXPECT_FALSE(guard.run(fail, report));
    EXPECT_EQ(payloadCalls, 1);
    EXPECT_EQ(reportCalls, 1);
  }
  fast_lio::FrontendPublicationGuard healthyGuard;
  int reports = 0;
  EXPECT_TRUE(healthyGuard.run([] {}, [&](const char *) { ++reports; }));
  EXPECT_FALSE(healthyGuard.faulted());
  EXPECT_EQ(reports, 0);
}

TEST(FrontendObservation, EventSecondsConversionRejectsInvalidBounds)
{
  std::int64_t stamp = 0;
  EXPECT_TRUE(fast_lio::frontendSecondsStamp(100.5, stamp));
  EXPECT_EQ(stamp, 100500000000);
  EXPECT_TRUE(fast_lio::frontendSecondsStamp(-0.5, stamp));
  EXPECT_EQ(stamp, -500000000);
  EXPECT_FALSE(fast_lio::frontendSecondsStamp(std::numeric_limits<double>::quiet_NaN(), stamp));
  EXPECT_FALSE(fast_lio::frontendSecondsStamp(std::numeric_limits<double>::infinity(), stamp));
  EXPECT_FALSE(fast_lio::frontendSecondsStamp(2147483648.0, stamp));
  EXPECT_EQ(stamp, 0);
}

TEST(FrontendObservation, BoundedUtf8Identifiers)
{
  EXPECT_TRUE(fast_lio::frontendIdentifier("sensor", 48));
  EXPECT_TRUE(fast_lio::frontendIdentifier("\xe4\xbc\xa0\xe6\x84\x9f", 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier("", 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier(std::string(49, 'x'), 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier(std::string("a\0b", 3), 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier("\xc0\x80", 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier("\xed\xa0\x80", 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier("\xf4\x90\x80\x80", 48));
  EXPECT_FALSE(fast_lio::frontendIdentifier("\xe4\xbc", 48));
}

TEST(FrontendObservation, HealthyAndUnknownInputs)
{
  auto facts = healthy();
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 1);
  EXPECT_EQ(fast_lio::frontendVerdict(facts).axes, 63);
  facts.contextKnown = false;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 0);
  facts = healthy();
  facts.stampKnown = false;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).axes, 0);
  for (const auto bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    for (std::size_t i = 0; i < facts.pose.size(); ++i) {
      facts = healthy();
      facts.pose[i] = bad;
      EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 0);
    }
  }
  facts = healthy();
  facts.pose[6] = 2.0;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 0);
}

TEST(FrontendObservation, GuardsCannotInventAxesOrLost)
{
  auto facts = healthy();
  facts.staticHold = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).state, 2);
  EXPECT_EQ(fast_lio::frontendVerdict(facts).substate, 4);
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 1);
  facts.degraded = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 0);
  facts = healthy();
  facts.levelling = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).state, fast_lio::frontendWire(fast_lio::FrontendState::kDegraded));
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, fast_lio::frontendWire(fast_lio::FrontendValidity::kUnknown));
  EXPECT_EQ(fast_lio::frontendVerdict(facts).axes, 0);
  facts = healthy();
  facts.blind = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).substate, 2);
  EXPECT_EQ(fast_lio::frontendVerdict(facts).axes, 0);
  facts.blind = false;
  facts.verifying = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).substate, 3);
  facts.runaway = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).substate, 5);
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 0);
  facts.lost = true;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 3);
  facts.completed = false;
  EXPECT_EQ(fast_lio::frontendVerdict(facts).validity, 0);
  EXPECT_EQ(fast_lio::frontendVerdict(facts).axes, 0);
}

TEST(FrontendObservation, ProgressSeparatesCompletionOutputAndIdle)
{
  fast_lio::FrontendProgress progress;
  progress.admit(100, true);
  auto facts = healthy();
  progress.finish(fast_lio::frontendVerdict(facts));
  facts.lost = true;
  progress.finish(fast_lio::frontendVerdict(facts));
  EXPECT_EQ(progress.completion, 2u);
  EXPECT_EQ(progress.output, 1u);
  facts.completed = false;
  const auto idle = fast_lio::frontendVerdict(facts);
  EXPECT_TRUE(progress.idleDue(100, 500, idle));
  EXPECT_FALSE(progress.idleDue(599, 500, idle));
  EXPECT_TRUE(progress.idleDue(600, 500, idle));
  EXPECT_FALSE(progress.idleDue(0, 500, idle));
  EXPECT_EQ(progress.completion, 2u);
  facts.lost = false;
  EXPECT_TRUE(progress.idleDue(601, 500, fast_lio::frontendVerdict(facts)));
  progress.input = std::numeric_limits<std::uint64_t>::max();
  progress.admit(602, true);
  EXPECT_FALSE(progress.valid);
  EXPECT_EQ(progress.input, std::numeric_limits<std::uint64_t>::max());
}

TEST(FrontendObservation, ExactStampAndHashGoldenVector)
{
  std::int64_t stamp = 0;
  EXPECT_TRUE(fast_lio::frontendStamp(-1, 999999999, stamp));
  EXPECT_EQ(stamp, -1);
  EXPECT_FALSE(fast_lio::frontendStamp(0, 1000000000, stamp));
  std::array<std::uint8_t, 16> instance{};
  for (std::size_t i = 0; i < instance.size(); ++i)
    instance[i] = static_cast<std::uint8_t>(i);
  std::array<std::uint8_t, 32> digest{};
  ASSERT_TRUE(fast_lio::frontendObservationId(instance, true, 7, -123456789, true, "sensor", digest));
  // Independently generated by Python hashlib, test/frontend_identity_vector.py.
  const std::string golden = "473a67e07bf10233a344f6f6728c053671149c3ec3cf5e747fbf4d2bfa875a16";
  const char hex[] = "0123456789abcdef";
  std::string actual;
  for (const auto byte : digest) {
    actual += hex[byte >> 4];
    actual += hex[byte & 15];
  }
  EXPECT_EQ(actual, golden);
  const auto original = digest;
  instance[0] = 99;  // another activation never aliases this completed scan
  ASSERT_TRUE(fast_lio::frontendObservationId(instance, true, 7, -123456789, true, "sensor", digest));
  EXPECT_NE(digest, original);
  instance[0] = 0;
  ASSERT_TRUE(fast_lio::frontendObservationId(instance, true, 8, -123456789, true, "sensor", digest));
  EXPECT_NE(digest, original);
  EXPECT_FALSE(fast_lio::frontendObservationId(instance, false, 7, 0, true, "sensor", digest));
  EXPECT_TRUE(std::all_of(digest.begin(), digest.end(), [](std::uint8_t value) { return value == 0; }));
  EXPECT_FALSE(fast_lio::frontendObservationId(instance, true, 0, 0, true, "sensor", digest));
  EXPECT_FALSE(fast_lio::frontendObservationId(instance, true, 7, 0, false, "sensor", digest));
  EXPECT_FALSE(fast_lio::frontendObservationId(instance, true, 7, 0, true, "", digest));
  EXPECT_FALSE(fast_lio::frontendObservationId(instance, true, 7, 0, true, std::string(49, 'x'), digest));
}
}  // namespace

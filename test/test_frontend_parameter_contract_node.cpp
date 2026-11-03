// The parameter contract of the legacy front-end node (laserMapping S0, design checklist step 3).
//
// Every case activates the real LaserMappingNode (fastlio_mapping_test_component, taken through class_loader and
// its NodeFactory like the other node fixtures) with one configuration, in a forked child of its own: the
// component's globals outlive a node, so one process activates it once. The child records what the node did with
// the configuration:
//   - the configuration itself: the profile file (path in the package and sha256) and every override;
//   - whether the constructor returned or threw, with the exception's type and text;
//   - every line the constructor logged through rcutils at INFO or above, from the constructing thread;
//   - for a constructed node, every declared parameter as the node's own parameter services report it: name, type,
//     value and descriptor (read-only flag, description, constraints, ranges);
//   - the outcome of a fixed set of runtime set requests, and the value each parameter has afterwards.
// The record is compared, line for line and character for character, with the golden the same executable generated
// from the base build (test/frontend_contract/README.md holds the command). Doubles are written with 17 significant
// digits, which round-trips, and NaN with its bit pattern: the comparison is bitwise, with no tolerance. A
// generation run records every case twice and writes a golden only when both records agree.
//
// Masked: the package's source directory in captured text (ROOT_DIR, written as <ROOT_DIR>/): the directory the
// package was configured in differs between the cloud build and CI. Nothing else is masked. Not recorded: what the
// constructor prints to std::cout, and log lines below INFO.
//
// The Defaults case stores its whole parameter table. Every other case stores only the parameter lines that differ
// from the Defaults table; the comparison rebuilds each case's full table from the Defaults golden and its own lines,
// so a changed default shows in Defaults and in every case that does not set that parameter.
//
// The goldens apply to x86_64 only. On other architectures CMake registers the CTest entry as skipped, and the
// executable, if run there in comparison mode, fails with that statement rather than pass.

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <class_loader/class_loader.hpp>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <exception>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <openssl/evp.h>
#include <optional>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rcutils/logging.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <typeinfo>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{

constexpr const char * kLaserMappingFactory = "rclcpp_components::NodeFactoryTemplate<LaserMappingNode>";
constexpr const char * kFixture = "parameter";
constexpr const char * kDefaultsCase = "defaults";
constexpr unsigned kChildAlarmSeconds = 120;
constexpr std::chrono::seconds kServiceWait{20};

#if defined(__x86_64__)
constexpr const char * kArch = "x86_64";
#elif defined(__aarch64__)
constexpr const char * kArch = "aarch64";
#else
constexpr const char * kArch = "other";
#endif

// ---------------------------------------------------------------------------
// Text of a record
// ---------------------------------------------------------------------------

std::string formatDouble(double value)
{
  char text[64];
  if (std::isnan(value)) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    std::snprintf(text, sizeof(text), "nan:0x%016llx", static_cast<unsigned long long>(bits));
  } else if (std::isinf(value)) {
    std::snprintf(text, sizeof(text), "%s", value > 0.0 ? "inf" : "-inf");
  } else {
    std::snprintf(text, sizeof(text), "%.17g", value);
  }
  return text;
}

// The package's source directory differs between builds: written as <ROOT_DIR>/ wherever it appears.
std::string mask(std::string text)
{
  const std::string root = ROOT_DIR;
  for (std::size_t at = text.find(root); !root.empty() && at != std::string::npos; at = text.find(root, at))
    text.replace(at, root.size(), "<ROOT_DIR>/");
  return text;
}

std::string quote(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (byte < 0x20 || byte >= 0x7f) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\x%02x", static_cast<unsigned>(byte));
      out += escaped;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string formatValue(const rclcpp::Parameter & parameter)
{
  std::ostringstream out;
  switch (parameter.get_type()) {
    case rclcpp::ParameterType::PARAMETER_NOT_SET:
      return "unset";
    case rclcpp::ParameterType::PARAMETER_BOOL:
      return parameter.as_bool() ? "true" : "false";
    case rclcpp::ParameterType::PARAMETER_INTEGER:
      return std::to_string(parameter.as_int());
    case rclcpp::ParameterType::PARAMETER_DOUBLE:
      return formatDouble(parameter.as_double());
    case rclcpp::ParameterType::PARAMETER_STRING:
      return quote(mask(parameter.as_string()));
    case rclcpp::ParameterType::PARAMETER_BYTE_ARRAY: {
      out << '[';
      const auto & values = parameter.as_byte_array();
      for (std::size_t i = 0; i < values.size(); ++i)
        out << (i ? "," : "") << static_cast<unsigned>(values[i]);
      out << ']';
      return out.str();
    }
    case rclcpp::ParameterType::PARAMETER_BOOL_ARRAY: {
      out << '[';
      const auto & values = parameter.as_bool_array();
      for (std::size_t i = 0; i < values.size(); ++i)
        out << (i ? "," : "") << (values[i] ? "true" : "false");
      out << ']';
      return out.str();
    }
    case rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY: {
      out << '[';
      const auto & values = parameter.as_integer_array();
      for (std::size_t i = 0; i < values.size(); ++i)
        out << (i ? "," : "") << values[i];
      out << ']';
      return out.str();
    }
    case rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY: {
      out << '[';
      const auto & values = parameter.as_double_array();
      for (std::size_t i = 0; i < values.size(); ++i)
        out << (i ? "," : "") << formatDouble(values[i]);
      out << ']';
      return out.str();
    }
    case rclcpp::ParameterType::PARAMETER_STRING_ARRAY: {
      out << '[';
      const auto & values = parameter.as_string_array();
      for (std::size_t i = 0; i < values.size(); ++i)
        out << (i ? "," : "") << quote(mask(values[i]));
      out << ']';
      return out.str();
    }
  }
  return "unknown_type";
}

std::string formatParameter(const rclcpp::Parameter & parameter)
{
  return "type=" + parameter.get_type_name() + " value=" + formatValue(parameter);
}

std::string formatDescriptor(const rcl_interfaces::msg::ParameterDescriptor & descriptor)
{
  std::ostringstream out;
  out << "read_only=" << (descriptor.read_only ? 1 : 0)
      << " descriptor_type=" << static_cast<unsigned>(descriptor.type);
  if (!descriptor.description.empty())
    out << " description=" << quote(descriptor.description);
  if (!descriptor.additional_constraints.empty())
    out << " constraints=" << quote(descriptor.additional_constraints);
  for (const auto & range : descriptor.floating_point_range)
    out << " fp_range=[" << formatDouble(range.from_value) << ',' << formatDouble(range.to_value) << ','
        << formatDouble(range.step) << ']';
  for (const auto & range : descriptor.integer_range)
    out << " int_range=[" << range.from_value << ',' << range.to_value << ',' << range.step << ']';
  return out.str();
}

const char * severityName(int severity)
{
  switch (severity) {
    case RCUTILS_LOG_SEVERITY_DEBUG:
      return "DEBUG";
    case RCUTILS_LOG_SEVERITY_INFO:
      return "INFO";
    case RCUTILS_LOG_SEVERITY_WARN:
      return "WARN";
    case RCUTILS_LOG_SEVERITY_ERROR:
      return "ERROR";
    case RCUTILS_LOG_SEVERITY_FATAL:
      return "FATAL";
    default:
      return "UNSET";
  }
}

std::string demangledType(const std::exception & error)
{
  int status = 0;
  char * name = abi::__cxa_demangle(typeid(error).name(), nullptr, nullptr, &status);
  const std::string text = status == 0 && name != nullptr ? std::string(name) : std::string(typeid(error).name());
  std::free(name);
  return text;
}

std::string hex(const unsigned char * bytes, std::size_t size)
{
  std::string out;
  char digits[3];
  for (std::size_t i = 0; i < size; ++i) {
    std::snprintf(digits, sizeof(digits), "%02x", static_cast<unsigned>(bytes[i]));
    out += digits;
  }
  return out;
}

bool readFile(const std::string & path, std::string * out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return !in.bad();
}

std::string sha256Hex(const std::string & bytes)
{
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest, &length, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("sha256 unavailable");
  return hex(digest, length);
}

std::vector<std::string> splitLines(const std::string & text)
{
  std::vector<std::string> lines;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  return lines;
}

// ---------------------------------------------------------------------------
// Log capture (child only)
// ---------------------------------------------------------------------------

struct CapturedLog
{
  int severity;
  std::string logger;
  std::string text;
};

std::mutex g_log_mutex;
std::vector<CapturedLog> g_logs;
bool g_capturing = false;
std::thread::id g_capture_thread;
rcutils_logging_output_handler_t g_previous_handler = nullptr;

// Records what the constructing thread logs and hands every line on to the handler rclcpp installed.
void captureLogLine(const rcutils_log_location_t * location,
                    int severity,
                    const char * name,
                    rcutils_time_point_value_t timestamp,
                    const char * format,
                    va_list * args)
{
  {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_capturing && severity >= RCUTILS_LOG_SEVERITY_INFO && std::this_thread::get_id() == g_capture_thread) {
      va_list copy;
      va_copy(copy, *args);
      char text[8192];
      const int length = std::vsnprintf(text, sizeof(text), format, copy);
      va_end(copy);
      g_logs.push_back({severity, name != nullptr ? name : "", length < 0 ? std::string("<unformattable>") : text});
    }
  }
  if (g_previous_handler != nullptr)
    g_previous_handler(location, severity, name, timestamp, format, args);
}

void startLogCapture()
{
  std::lock_guard<std::mutex> lock(g_log_mutex);
  g_logs.clear();
  g_capture_thread = std::this_thread::get_id();
  g_capturing = true;
  g_previous_handler = rcutils_logging_get_output_handler();
  rcutils_logging_set_output_handler(captureLogLine);
}

std::vector<CapturedLog> stopLogCapture()
{
  std::lock_guard<std::mutex> lock(g_log_mutex);
  g_capturing = false;
  rcutils_logging_set_output_handler(g_previous_handler);
  return g_logs;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

enum class Expect
{
  kConstructed,
  kRejected,
};

struct ParameterCase
{
  std::string name;
  Expect expect = Expect::kConstructed;
  std::string params_file;                   // in the package source directory; empty: overrides only
  std::vector<rclcpp::Parameter> overrides;  // NodeOptions::parameter_overrides, on top of the profile
  std::vector<rclcpp::Parameter> sets;       // runtime set requests, one per request, after the table was read
  // A profile case's own check that its file reached the node: the parameter and the value text it must have.
  std::vector<std::pair<std::string, std::string>> applied;
};

void setOverride(ParameterCase & c, const rclcpp::Parameter & parameter)
{
  for (auto & existing : c.overrides) {
    if (existing.get_name() == parameter.get_name()) {
      existing = parameter;
      return;
    }
  }
  c.overrides.push_back(parameter);
}

// The two extrinsics have empty defaults, and the constructor indexes both without a size check
// (VEC_FROM_ARRAY(extrinT), MAT_FROM_ARRAY(extrinR)): with the declared defaults it reads past two empty vectors.
// That is not a configuration a case can record, so every case that does not take them from a profile gives the
// identity, as every node fixture of this package does.
ParameterCase overridesCase(const std::string & name, Expect expect)
{
  ParameterCase c;
  c.name = name;
  c.expect = expect;
  setOverride(c, rclcpp::Parameter("mapping.extrinsic_T", std::vector<double>{0.0, 0.0, 0.0}));
  setOverride(
    c, rclcpp::Parameter("mapping.extrinsic_R", std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  return c;
}

ParameterCase profileCase(const std::string & name, const std::string & file, const std::string & lidar_topic)
{
  ParameterCase c;
  c.name = name;
  c.expect = Expect::kConstructed;
  c.params_file = file;
  c.applied.emplace_back("common.lid_topic", quote(lidar_topic));
  return c;
}

// ---------------------------------------------------------------------------
// The child: one activation, one configuration
// ---------------------------------------------------------------------------

template<class T>
T complete(rclcpp::executors::SingleThreadedExecutor & executor, std::shared_future<T> future, const std::string & what)
{
  if (executor.spin_until_future_complete(future, kServiceWait) != rclcpp::FutureReturnCode::SUCCESS)
    throw std::runtime_error(what + " did not complete within the service wait");
  return future.get();
}

void appendTable(std::ostringstream & record,
                 rclcpp::executors::SingleThreadedExecutor & executor,
                 rclcpp::AsyncParametersClient & client)
{
  auto names = complete(executor, client.list_parameters({}, 0), "list_parameters").names;
  std::sort(names.begin(), names.end());
  const auto values = complete(executor, client.get_parameters(names), "get_parameters");
  const auto descriptors = complete(executor, client.describe_parameters(names), "describe_parameters");
  if (values.size() != names.size() || descriptors.size() != names.size())
    throw std::runtime_error("the parameter services answered for a different number of parameters");
  record << "parameters " << names.size() << '\n';
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (values[i].get_name() != names[i])
      throw std::runtime_error("get_parameters answered " + values[i].get_name() + " for " + names[i]);
    record << "param " << names[i] << ' ' << formatParameter(values[i]) << ' ' << formatDescriptor(descriptors[i])
           << '\n';
  }
}

void appendSets(std::ostringstream & record,
                rclcpp::executors::SingleThreadedExecutor & executor,
                rclcpp::AsyncParametersClient & client,
                const std::vector<rclcpp::Parameter> & sets)
{
  for (const auto & request : sets) {
    const auto results = complete(executor, client.set_parameters({request}), "set_parameters");
    if (results.size() != 1)
      throw std::runtime_error("set_parameters answered " + std::to_string(results.size()) + " results for one");
    record << "set " << request.get_name() << ' ' << formatParameter(request) << " -> "
           << (results.front().successful ? std::string("ok") : "rejected " + quote(mask(results.front().reason)))
           << '\n';
    const auto after = complete(executor, client.get_parameters({request.get_name()}), "get_parameters");
    record << "after_set " << request.get_name() << ' ';
    if (after.empty())
      record << "undeclared\n";
    else
      record << formatParameter(after.front()) << '\n';
  }
}

std::string recordCase(const ParameterCase & c)
{
  // The node reads these at construction; the configuration under test is only what the case gives.
  for (const char * name : {"FLIO_ASYNC_MAP", "FLIO_OFFLINE_STEP", "FLIO_LIDAR_QDEPTH", "FLIO_IKD_PROFILE"})
    ::unsetenv(name);
  std::ostringstream record;
  record << "case " << c.name << '\n';
  if (c.params_file.empty()) {
    record << "config overrides-only\n";
  } else {
    std::string bytes;
    if (!readFile(std::string(ROOT_DIR) + c.params_file, &bytes))
      throw std::runtime_error("cannot read the profile " + c.params_file);
    record << "config " << c.params_file << " sha256 " << sha256Hex(bytes) << '\n';
  }
  auto overrides = c.overrides;
  std::sort(overrides.begin(), overrides.end(), [](const rclcpp::Parameter & a, const rclcpp::Parameter & b) {
    return a.get_name() < b.get_name();
  });
  record << "overrides " << overrides.size() << '\n';
  for (const auto & parameter : overrides)
    record << "override " << parameter.get_name() << ' ' << formatParameter(parameter) << '\n';

  rclcpp::init(0, nullptr);
  auto loader = std::make_unique<class_loader::ClassLoader>(FASTLIO_TEST_COMPONENT_PATH);
  const auto classes = loader->getAvailableClasses<rclcpp_components::NodeFactory>();
  if (std::find(classes.begin(), classes.end(), kLaserMappingFactory) == classes.end())
    throw std::runtime_error("LaserMappingNode factory not found");
  auto factory = loader->createInstance<rclcpp_components::NodeFactory>(kLaserMappingFactory);
  rclcpp::NodeOptions options;
  if (!c.params_file.empty())
    options.arguments({"--ros-args", "--params-file", std::string(ROOT_DIR) + c.params_file});
  options.parameter_overrides(c.overrides);

  std::optional<rclcpp_components::NodeInstanceWrapper> subject;
  std::string outcome;
  startLogCapture();
  try {
    subject.emplace(factory->create_node_instance(options));
    outcome = "constructed";
  } catch (const std::exception & error) {
    outcome = "threw " + demangledType(error) + ": " + quote(mask(error.what()));
  } catch (...) {
    outcome = "threw unknown";
  }
  const auto logs = stopLogCapture();
  record << "outcome " << outcome << '\n';
  for (const auto & line : logs)
    record << "log " << severityName(line.severity) << ' ' << line.logger << ' ' << quote(mask(line.text)) << '\n';

  if (subject) {
    auto observer = std::make_shared<rclcpp::Node>("frontend_parameter_contract_observer");
    auto client = std::make_shared<rclcpp::AsyncParametersClient>(observer, "laser_mapping");
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(subject->get_node_base_interface());
    executor.add_node(observer);
    if (!client->wait_for_service(kServiceWait))
      throw std::runtime_error("the node's parameter services never became available");
    appendTable(record, executor, *client);
    appendSets(record, executor, *client, c.sets);
    executor.remove_node(observer);
    executor.remove_node(subject->get_node_base_interface());
    client.reset();
    observer.reset();
    subject.reset();  // the production destructor stops and joins the node's workers
  }
  factory.reset();
  rclcpp::shutdown();
  loader.reset();
  return record.str();
}

// ---------------------------------------------------------------------------
// The parent: fork, collect, compare
// ---------------------------------------------------------------------------

void writeAll(int fd, const std::string & text)
{
  const char * bytes = text.data();
  std::size_t size = text.size();
  while (size > 0) {
    const ssize_t written = ::write(fd, bytes, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return;
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
}

bool readAll(int fd, std::string * out)
{
  char buffer[65536];
  for (;;) {
    const ssize_t got = ::read(fd, buffer, sizeof(buffer));
    if (got == 0)
      return true;
    if (got < 0 && errno == EINTR)
      continue;
    if (got < 0)
      return false;
    out->append(buffer, static_cast<std::size_t>(got));
  }
}

struct ChildRun
{
  int status = -1;
  std::string record;
  std::string error;

  bool ok() const { return error.empty() && WIFEXITED(status) && WEXITSTATUS(status) == 0; }
  std::string describe() const
  {
    std::ostringstream out;
    out << "child status=" << status;
    if (WIFSIGNALED(status))
      out << " signal=" << WTERMSIG(status);
    if (WIFEXITED(status))
      out << " exit=" << WEXITSTATUS(status);
    out << " error='" << error << "'";
    return out.str();
  }
};

ChildRun runInChild(const std::function<std::string()> & body)
{
  ChildRun run;
  int fds[2];
  if (::pipe(fds) != 0) {
    run.error = "pipe failed";
    return run;
  }
  std::fflush(stdout);
  std::fflush(stderr);
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    run.error = "fork failed";
    return run;
  }
  if (pid == 0) {
    ::close(fds[0]);
    ::alarm(kChildAlarmSeconds);
    std::string text;
    int code = 0;
    try {
      text = "@@record\n" + body();
    } catch (const std::exception & error) {
      text = std::string("@@error ") + error.what() + "\n";
      code = 3;
    } catch (...) {
      text = "@@error unknown exception\n";
      code = 3;
    }
    writeAll(fds[1], text);
    ::close(fds[1]);
    ::_exit(code);
  }
  ::close(fds[1]);
  std::string output;
  const bool read_ok = readAll(fds[0], &output);
  ::close(fds[0]);
  ::waitpid(pid, &run.status, 0);
  const std::string record_tag = "@@record\n";
  const std::string error_tag = "@@error ";
  if (!read_ok)
    run.error = "pipe read failed";
  else if (output.compare(0, record_tag.size(), record_tag) == 0)
    run.record = output.substr(record_tag.size());
  else if (output.compare(0, error_tag.size(), error_tag) == 0)
    run.error = output.substr(error_tag.size());
  else
    run.error = "the child wrote no record";
  return run;
}

// A record split around its parameter table: the lines before it (up to "parameters N"), the table by name, and the
// lines after it.
struct SplitRecord
{
  std::vector<std::string> before;
  std::map<std::string, std::string> table;
  std::vector<std::string> absent;  // names a stored delta says the case does not have
  std::vector<std::string> after;
};

std::string parameterName(const std::string & line)
{
  const std::size_t start = line.find(' ') + 1;
  return line.substr(start, line.find(' ', start) - start);
}

SplitRecord splitRecord(const std::string & text)
{
  SplitRecord split;
  bool seen_table = false;
  for (const auto & line : splitLines(text)) {
    if (line.rfind("# ", 0) == 0)
      continue;  // a golden's header
    if (line.rfind("param ", 0) == 0) {
      split.table[parameterName(line)] = line;
      seen_table = true;
    } else if (line.rfind("param_absent ", 0) == 0) {
      split.absent.push_back(line.substr(std::string("param_absent ").size()));
      seen_table = true;
    } else if (seen_table || line.rfind("set ", 0) == 0 || line.rfind("after_set ", 0) == 0) {
      seen_table = true;
      split.after.push_back(line);
    } else {
      split.before.push_back(line);
    }
  }
  return split;
}

std::string joinRecord(const SplitRecord & split)
{
  std::string out;
  for (const auto & line : split.before)
    out += line + '\n';
  for (const auto & entry : split.table)
    out += entry.second + '\n';
  for (const auto & name : split.absent)
    out += "param_absent " + name + '\n';
  for (const auto & line : split.after)
    out += line + '\n';
  return out;
}

// What a case stores: its table as the lines that differ from the reference table.
std::string toStored(const std::string & record, const std::map<std::string, std::string> & reference)
{
  SplitRecord split = splitRecord(record);
  std::map<std::string, std::string> delta;
  for (const auto & entry : split.table) {
    const auto it = reference.find(entry.first);
    if (it == reference.end() || it->second != entry.second)
      delta.insert(entry);
  }
  // A constructed node that lacks a parameter of the reference table says so; a node that was not constructed has
  // no table at all.
  if (!split.table.empty()) {
    for (const auto & entry : reference)
      if (split.table.count(entry.first) == 0)
        split.absent.push_back(entry.first);
  }
  split.table = delta;
  return joinRecord(split);
}

// What a stored case expands to: the reference table with the case's own lines applied. A record has a table when it
// has the "parameters N" line; a case whose table equals the reference stores no table line at all.
std::string expandStored(const std::string & stored, const std::map<std::string, std::string> & reference)
{
  SplitRecord split = splitRecord(stored);
  const bool has_table = std::any_of(split.before.begin(), split.before.end(), [](const std::string & line) {
    return line.rfind("parameters ", 0) == 0;
  });
  if (!has_table)
    return joinRecord(split);  // no table: the node was not constructed
  std::map<std::string, std::string> table = reference;
  for (const auto & entry : split.table)
    table[entry.first] = entry.second;
  for (const auto & name : split.absent)
    table.erase(name);
  split.table = table;
  split.absent.clear();
  return joinRecord(split);
}

bool makeDirectories(const std::string & path)
{
  std::string partial;
  std::istringstream parts(path);
  if (!path.empty() && path[0] == '/')
    partial = "/";
  for (std::string part; std::getline(parts, part, '/');) {
    if (part.empty())
      continue;
    partial += part + "/";
    if (::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
      return false;
  }
  return true;
}

bool writeFile(const std::string & directory, const std::string & name, const std::string & text)
{
  if (!makeDirectories(directory))
    return false;
  std::ofstream out(directory + "/" + name, std::ios::binary | std::ios::trunc);
  out << text;
  return static_cast<bool>(out);
}

std::string goldenOut()
{
  const char * value = std::getenv("FASTLIO_CONTRACT_GOLDEN_OUT");
  return value != nullptr ? std::string(value) : std::string();
}

std::string goldenDir()
{
  return std::string(ROOT_DIR) + "test/frontend_contract/golden/" + kFixture;
}

std::string goldenHeader(const std::string & case_name)
{
  const char * sha = std::getenv("FASTLIO_CONTRACT_SOURCE_SHA");
  std::ostringstream out;
  out << "# frontend contract golden: " << kFixture << '/' << case_name << '\n'
      << "# fork_sha: " << (sha != nullptr ? sha : "") << '\n'
      << "# arch: " << kArch << '\n'
      << "# ros_distro: " << FASTLIO_CONTRACT_ROS_DISTRO << '\n'
      << "# build: " << FASTLIO_CONTRACT_BUILD << '\n';
  return out.str();
}

std::string headerValue(const std::string & golden, const std::string & key)
{
  const std::string prefix = "# " + key + ": ";
  for (const auto & line : splitLines(golden))
    if (line.rfind(prefix, 0) == 0)
      return line.substr(prefix.size());
  return "";
}

std::string jsonString(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (byte < 0x20) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(byte));
      out += escaped;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

struct Comparison
{
  std::string result;  // equal, different, generated, no_golden, not_comparable
  std::string detail;
  std::size_t line = 0;
  std::string expected, actual;
};

Comparison compareLines(const std::string & expected, const std::string & actual)
{
  Comparison comparison;
  const auto want = splitLines(expected);
  const auto got = splitLines(actual);
  for (std::size_t i = 0; i < std::max(want.size(), got.size()); ++i) {
    const std::string a = i < want.size() ? want[i] : "<end of record>";
    const std::string b = i < got.size() ? got[i] : "<end of record>";
    if (a != b) {
      comparison.result = "different";
      comparison.line = i + 1;
      comparison.expected = a;
      comparison.actual = b;
      return comparison;
    }
  }
  comparison.result = "equal";
  return comparison;
}

void writeReceipt(const std::string & case_name,
                  const std::string & mode,
                  const ChildRun & run,
                  const Comparison & comparison,
                  const std::string & golden_path,
                  const std::string & golden_text)
{
  std::ostringstream out;
  out << "{\"fixture\":" << jsonString(kFixture) << ",\"case\":" << jsonString(case_name)
      << ",\"mode\":" << jsonString(mode) << ",\"arch\":" << jsonString(kArch)
      << ",\"ros_distro\":" << jsonString(FASTLIO_CONTRACT_ROS_DISTRO)
      << ",\"build\":" << jsonString(FASTLIO_CONTRACT_BUILD) << ",\"child\":" << jsonString(run.describe())
      << ",\"record_lines\":" << splitLines(run.record).size()
      << ",\"record_sha256\":" << jsonString(sha256Hex(run.record)) << ",\"golden\":" << jsonString(mask(golden_path))
      << ",\"golden_sha256\":" << jsonString(golden_text.empty() ? std::string() : sha256Hex(golden_text))
      << ",\"golden_fork_sha\":" << jsonString(headerValue(golden_text, "fork_sha"))
      << ",\"golden_build\":" << jsonString(headerValue(golden_text, "build"))
      << ",\"result\":" << jsonString(comparison.result) << ",\"detail\":" << jsonString(comparison.detail)
      << ",\"first_difference_line\":" << comparison.line << ",\"expected\":" << jsonString(comparison.expected)
      << ",\"actual\":" << jsonString(comparison.actual) << "}\n";
  const std::string name = case_name + ".receipt.json";
  writeFile(std::string(FASTLIO_CONTRACT_RESULTS) + "/frontend_contract/" + kFixture, name, out.str());
  const std::string generate = goldenOut();
  if (!generate.empty())
    writeFile(generate + "/receipts/" + kFixture, name, out.str());
}

// The Defaults record of this build, for generating the other cases' stored lines: the Defaults case leaves it here
// when it runs first; otherwise the first case that needs it runs that child once.
std::optional<std::string> & defaultsRecordCache()
{
  static std::optional<std::string> cache;
  return cache;
}

ParameterCase defaultsCase();

std::map<std::string, std::string> referenceTable(const std::string & case_name, bool generate, std::string * error)
{
  if (case_name == kDefaultsCase)
    return {};
  std::string defaults;
  if (generate) {
    auto & cache = defaultsRecordCache();
    if (!cache) {
      const ChildRun run = runInChild([] { return recordCase(defaultsCase()); });
      if (!run.ok()) {
        *error = "the Defaults child for the reference table failed: " + run.describe();
        return {};
      }
      cache = run.record;
    }
    defaults = *cache;
  } else if (!readFile(goldenDir() + "/" + kDefaultsCase + ".golden", &defaults)) {
    *error = "no Defaults golden to rebuild this case's table from: " + mask(goldenDir()) + "/defaults.golden";
    return {};
  }
  return splitRecord(defaults).table;
}

void checkCase(const ParameterCase & c)
{
  const std::string generate = goldenOut();
  if (generate.empty() && std::string(kArch) != "x86_64") {
    FAIL() << "golden 只适用于 x86_64: the front-end contract goldens were generated on x86_64 and this build is "
           << kArch
           << "; the comparison is not run here (CMake registers this entry as skipped on other architectures)";
  }
  const ChildRun run = runInChild([&c] { return recordCase(c); });
  ASSERT_TRUE(run.ok()) << c.name << ": " << run.describe();
  if (c.name == kDefaultsCase)
    defaultsRecordCache() = run.record;

  // The case reached what it is about: a constructed node or a refused configuration, and a profile that applied.
  const bool constructed = run.record.find("\noutcome constructed\n") != std::string::npos;
  EXPECT_EQ(constructed, c.expect == Expect::kConstructed)
    << c.name << " was meant to be " << (c.expect == Expect::kConstructed ? "constructed" : "refused") << ":\n"
    << run.record;
  if (constructed)
    EXPECT_NE(run.record.find("\nparameters "), std::string::npos) << c.name << " has no parameter table";
  for (const auto & applied : c.applied) {
    const std::string prefix = "param " + applied.first + " type=string value=" + applied.second + " ";
    EXPECT_NE(run.record.find(prefix), std::string::npos)
      << c.name << ": the profile did not reach the node: no line starting '" << prefix << "'";
  }

  std::string error;
  const auto reference = referenceTable(c.name, !generate.empty(), &error);
  Comparison comparison;
  std::string golden_text;
  std::string golden_path = goldenDir() + "/" + c.name + ".golden";
  if (!error.empty()) {
    comparison.result = "error";
    comparison.detail = error;
  } else if (!generate.empty()) {
    // A golden is written only from a record two runs of this build agree on.
    const ChildRun again = runInChild([&c] { return recordCase(c); });
    if (!again.ok()) {
      comparison.result = "error";
      comparison.detail = "second run: " + again.describe();
    } else if (again.record != run.record) {
      comparison = compareLines(run.record, again.record);
      comparison.result = "nondeterministic";
      comparison.detail = "two runs of this build recorded different text";
      writeFile(std::string(FASTLIO_CONTRACT_RESULTS) + "/frontend_contract/" + kFixture,
                c.name + ".second.actual",
                again.record);
    } else {
      golden_path = generate + "/golden/" + kFixture + "/" + c.name + ".golden";
      golden_text = goldenHeader(c.name) + toStored(run.record, reference);
      const Comparison round_trip =
        compareLines(expandStored(golden_text, reference), joinRecord(splitRecord(run.record)));
      if (round_trip.result != "equal") {
        comparison = round_trip;
        comparison.result = "error";
        comparison.detail = "the stored golden does not expand to the record it was made from";
      } else if (!writeFile(generate + "/golden/" + kFixture, c.name + ".golden", golden_text)) {
        comparison.result = "error";
        comparison.detail = "cannot write " + golden_path;
      } else {
        comparison.result = "generated";
      }
    }
  } else if (!readFile(golden_path, &golden_text)) {
    comparison.result = "no_golden";
    comparison.detail = "no golden at " + mask(golden_path);
  } else if (headerValue(golden_text, "arch") != "x86_64" ||
             headerValue(golden_text, "ros_distro") != FASTLIO_CONTRACT_ROS_DISTRO) {
    comparison.result = "not_comparable";
    comparison.detail = "the golden was generated on " + headerValue(golden_text, "arch") + "/" +
                        headerValue(golden_text, "ros_distro") + ", this build is " + kArch + "/" +
                        FASTLIO_CONTRACT_ROS_DISTRO;
  } else {
    comparison = compareLines(expandStored(golden_text, reference), joinRecord(splitRecord(run.record)));
    if (comparison.result != "equal") {
      comparison.detail =
        "golden build: " + headerValue(golden_text, "build") + "; this build: " + FASTLIO_CONTRACT_BUILD;
      writeFile(
        std::string(FASTLIO_CONTRACT_RESULTS) + "/frontend_contract/" + kFixture, c.name + ".actual", run.record);
    }
  }
  writeReceipt(c.name, generate.empty() ? "compare" : "generate", run, comparison, golden_path, golden_text);
  if (!generate.empty()) {
    EXPECT_EQ(comparison.result, "generated") << comparison.detail;
    return;
  }
  EXPECT_EQ(comparison.result, "equal") << c.name << ": " << comparison.detail << "\n  line " << comparison.line
                                        << "\n  golden: " << comparison.expected << "\n  actual: " << comparison.actual;
}

// ---------------------------------------------------------------------------
// The cases (design checklist step 3)
// ---------------------------------------------------------------------------

// The declared defaults. Runtime requests: a read-only parameter, an ordinary value, the wrong type, an undeclared
// name.
ParameterCase defaultsCase()
{
  ParameterCase c = overridesCase(kDefaultsCase, Expect::kConstructed);
  c.sets = {rclcpp::Parameter("publish.reference_health", true),
            rclcpp::Parameter("filter_size_map", 0.4),
            rclcpp::Parameter("filter_size_map", std::string("0.4")),
            rclcpp::Parameter("frontend_contract.undeclared", 1)};
  return c;
}

TEST(FrontendParameterContract, Defaults)
{
  checkCase(defaultsCase());
}

// The three platform profiles. M20's front-end parameters are the parent repository's merged profile, kept here as
// a snapshot (test/frontend_contract/README.md says where it came from).
TEST(FrontendParameterContract, ProfileM20)
{
  checkCase(profileCase("profile_m20", "test/frontend_contract/m20_laser_mapping.yaml", "/lio/points"));
}

TEST(FrontendParameterContract, ProfileXswHesai)
{
  checkCase(profileCase("profile_xsw_hesai", "config/xsw_hesai.yaml", "/hesai210/original"));
}

TEST(FrontendParameterContract, ProfileGiant)
{
  checkCase(profileCase("profile_giant", "config/giant.yaml", "/rs_velo/points"));
}

// initial_pose_full_rpy_override and its deprecated name initial_pose_apply_roll_pitch, alone and together.
TEST(FrontendParameterContract, AliasNewNameOnly)
{
  ParameterCase c = overridesCase("alias_new_name_only", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("initial_pose_full_rpy_override", true));
  checkCase(c);
}

TEST(FrontendParameterContract, AliasDeprecatedNameOnly)
{
  ParameterCase c = overridesCase("alias_deprecated_name_only", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("initial_pose_apply_roll_pitch", true));
  checkCase(c);
}

TEST(FrontendParameterContract, AliasBothNamesSet)
{
  ParameterCase c = overridesCase("alias_both_names_set", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("initial_pose_full_rpy_override", true));
  setOverride(c, rclcpp::Parameter("initial_pose_apply_roll_pitch", true));
  checkCase(c);
}

TEST(FrontendParameterContract, AliasBothNamesDisagree)
{
  ParameterCase c = overridesCase("alias_both_names_disagree", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("initial_pose_full_rpy_override", false));
  setOverride(c, rclcpp::Parameter("initial_pose_apply_roll_pitch", true));
  checkCase(c);
}

// Type errors: an override whose type is not the declared one.
TEST(FrontendParameterContract, TypeStringForDouble)
{
  ParameterCase c = overridesCase("type_string_for_double", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("filter_size_map", std::string("0.5")));
  checkCase(c);
}

TEST(FrontendParameterContract, TypeIntegerForDouble)
{
  ParameterCase c = overridesCase("type_integer_for_double", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("filter_size_map", 1));
  checkCase(c);
}

TEST(FrontendParameterContract, TypeDoubleForInteger)
{
  ParameterCase c = overridesCase("type_double_for_integer", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("point_filter_num", 2.0));
  checkCase(c);
}

TEST(FrontendParameterContract, TypeIntegerForBool)
{
  ParameterCase c = overridesCase("type_integer_for_bool", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("feature_extract_enable", 1));
  checkCase(c);
}

TEST(FrontendParameterContract, TypeIntegerArrayForDoubleArray)
{
  ParameterCase c = overridesCase("type_integer_array_for_double_array", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("mapping.extrinsic_T", std::vector<int64_t>{0, 0, 0}));
  checkCase(c);
}

// Vector lengths. A short mapping.extrinsic_T is not a case: the constructor would read past it (see
// overridesCase).
TEST(FrontendParameterContract, VectorLengthLeverArm)
{
  ParameterCase c = overridesCase("vector_length_lever_arm", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("gravity_align_leg_lever_arm", std::vector<double>{0.1, 0.2}));
  checkCase(c);
}

TEST(FrontendParameterContract, VectorLengthWheelExtrinsic)
{
  ParameterCase c = overridesCase("vector_length_wheel_extrinsic", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("wheel_extrinsic_R", std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}));
  checkCase(c);
}

TEST(FrontendParameterContract, VectorLengthInitialPose)
{
  ParameterCase c = overridesCase("vector_length_initial_pose", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("initial_pose", std::vector<double>{1.0, 2.0, 3.0}));
  checkCase(c);
}

// Non-finite values.
TEST(FrontendParameterContract, LostRecoveryRequiresRecoveryBudget)
{
  ParameterCase c = overridesCase("lost_recovery_requires_budget", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("lost_recovery_en", true));
  setOverride(c, rclcpp::Parameter("reanchor_total_timeout_sec", 0.0));
  checkCase(c);
}

TEST(FrontendParameterContract, NanImuBacklog)
{
  ParameterCase c = overridesCase("nan_imu_backlog", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("max_imu_backlog_sec", std::numeric_limits<double>::quiet_NaN()));
  checkCase(c);
}

TEST(FrontendParameterContract, InfScanDuration)
{
  ParameterCase c = overridesCase("inf_scan_duration", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("preprocess.max_scan_duration_s", std::numeric_limits<double>::infinity()));
  checkCase(c);
}

TEST(FrontendParameterContract, InfImuCoverageGap)
{
  ParameterCase c = overridesCase("inf_imu_coverage_gap", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("reanchor_en", true));
  setOverride(c, rclcpp::Parameter("imu_coverage.max_gap_s", std::numeric_limits<double>::infinity()));
  checkCase(c);
}

TEST(FrontendParameterContract, NanExtrinsicElement)
{
  ParameterCase c = overridesCase("nan_extrinsic_element", Expect::kConstructed);
  setOverride(
    c,
    rclcpp::Parameter("mapping.extrinsic_T", std::vector<double>{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}));
  checkCase(c);
}

TEST(FrontendParameterContract, NanMapFilterSize)
{
  ParameterCase c = overridesCase("nan_map_filter_size", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("filter_size_map", std::numeric_limits<double>::quiet_NaN()));
  checkCase(c);
}

// The prior map: no file, an unknown alignment, a file that does not exist, a negative voxel. The map itself is
// loaded with the first scan, not by the constructor: these cases record what construction makes of the settings.
TEST(FrontendParameterContract, PriorMapEmptyWithRelocalization)
{
  ParameterCase c = overridesCase("prior_map_empty_relocalization", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("prior_map_pcd", std::string("")));
  setOverride(c, rclcpp::Parameter("prior_map_align", std::string("relocalization")));
  checkCase(c);
}

TEST(FrontendParameterContract, PriorMapUnknownAlignment)
{
  ParameterCase c = overridesCase("prior_map_unknown_alignment", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("prior_map_pcd", std::string("/nonexistent/fastlio_contract/prior.pcd")));
  setOverride(c, rclcpp::Parameter("prior_map_align", std::string("sideways")));
  checkCase(c);
}

TEST(FrontendParameterContract, PriorMapMissingFileWithRelocalization)
{
  ParameterCase c = overridesCase("prior_map_missing_file_relocalization", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("prior_map_pcd", std::string("/nonexistent/fastlio_contract/prior.pcd")));
  setOverride(c, rclcpp::Parameter("prior_map_align", std::string("relocalization")));
  checkCase(c);
}

TEST(FrontendParameterContract, PriorMapNegativeVoxel)
{
  ParameterCase c = overridesCase("prior_map_negative_voxel", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("prior_map_pcd", std::string("/nonexistent/fastlio_contract/prior.pcd")));
  setOverride(c, rclcpp::Parameter("prior_map_voxel", -1.0));
  checkCase(c);
}

// publish.reference_health is declared read-only: it is set at construction or not at all, and only together with
// the typed observation it needs.
TEST(FrontendParameterContract, ReadOnlyWithoutObservation)
{
  ParameterCase c = overridesCase("read_only_without_observation", Expect::kRejected);
  setOverride(c, rclcpp::Parameter("publish.reference_health", true));
  checkCase(c);
}

TEST(FrontendParameterContract, ReadOnlyWithObservation)
{
  ParameterCase c = overridesCase("read_only_with_observation", Expect::kConstructed);
  setOverride(c, rclcpp::Parameter("publish.reference_health", true));
  setOverride(c, rclcpp::Parameter("publish.frontend_observation", true));
  setOverride(c, rclcpp::Parameter("publish.frontend_clock_domain", std::string("fixture")));
  c.sets = {rclcpp::Parameter("publish.reference_health", false)};
  checkCase(c);
}

}  // namespace

#pragma once

#include <rclcpp/node.hpp>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>

namespace fast_lio
{
/// A front-end parameter that is declared with `declared` and later read with get_parameter_or, which falls back to
/// `fallback`. The two stay separate because the constructor always wrote them separately, and a few differ in
/// spelling or type (a float literal for a double, a string for a C string).
template<class T>
struct ParameterSpec
{
  const char * name;
  T declared;
  T fallback;
};

/// A front-end parameter whose declaration is also its read: the constructor keeps what declare_parameter returns,
/// or reads it back with get_parameter(name). read_only is the only descriptor field any of them sets.
template<class T>
struct DeclaredValueSpec
{
  const char * name;
  T declared;
  bool read_only;
};

/// declare_parameter<T>(name, declared) itself: a second declaration, or an override of another type, throws the
/// rclcpp exception it always threw.
template<class T>
void declareOnly(rclcpp::Node & node, const ParameterSpec<T> & spec)
{
  node.declare_parameter<T>(spec.name, spec.declared);
}

/// declare_parameter<T>(name, declared, descriptor), returning the declared value; the descriptor is the default one
/// with read_only set as the spec says.
template<class T>
T declareValue(rclcpp::Node & node, const DeclaredValueSpec<T> & spec)
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = spec.read_only;
  return node.declare_parameter<T>(spec.name, spec.declared, descriptor);
}

/// get_parameter_or<T>(name, destination, fallback): it always assigns destination, and a parameter of another
/// type throws.
template<class T>
void readInto(const rclcpp::Node & node, const ParameterSpec<T> & spec, T & destination)
{
  node.get_parameter_or<T>(spec.name, destination, spec.fallback);
}
}  // namespace fast_lio

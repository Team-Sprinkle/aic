#pragma once

#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aic::detail {

// A stationary arm can still be at the previous trial's pose. Match named
// positions as well as velocities; reject incomplete/nonfinite observations.
inline bool joints_ready_at_home(
    const std::vector<std::string>& names,
    const std::vector<double>& positions,
    const std::vector<double>& velocities,
    const std::vector<std::string>& home_names,
    const std::vector<double>& home_positions,
    double position_tolerance, double velocity_tolerance) {
  if (names.empty() || positions.size() != names.size() ||
      velocities.size() != names.size() || home_names.empty() ||
      home_names.size() != home_positions.size() ||
      !std::isfinite(position_tolerance) || position_tolerance <= 0.0 ||
      !std::isfinite(velocity_tolerance) || velocity_tolerance <= 0.0) {
    return false;
  }
  std::unordered_map<std::string, double> observed;
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty() || !std::isfinite(positions[i]) ||
        !std::isfinite(velocities[i]) ||
        std::fabs(velocities[i]) > velocity_tolerance ||
        !observed.emplace(names[i], positions[i]).second) {
      return false;
    }
  }
  std::unordered_set<std::string> expected;
  for (size_t i = 0; i < home_names.size(); ++i) {
    const auto found = observed.find(home_names[i]);
    if (!expected.insert(home_names[i]).second ||
        !std::isfinite(home_positions[i]) || found == observed.end() ||
        std::fabs(found->second - home_positions[i]) > position_tolerance) {
      return false;
    }
  }
  return true;
}

}  // namespace aic::detail

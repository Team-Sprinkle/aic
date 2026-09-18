#include "../src/joint_readiness.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

int main() {
  using aic::detail::joints_ready_at_home;
  const std::vector<std::string> home_names = {
      "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"};
  const std::vector<double> home = {-0.1597, -1.3542, -1.6648, -1.6933, 1.571, 1.411};
  auto names = home_names;
  names.push_back("gripper_joint");
  // Real nominal pre-command arm positions from the development recordings.
  const std::vector<double> nominal = {
      -0.15814, -1.35243, -1.68709, -1.67295, 1.57066, 1.41247, 0.00354};
  const std::vector<double> stationary(7, 0.0);
  int checks = 0;
  auto require = [&checks](bool condition, const char* description) {
    ++checks;
    if (!condition) {
      std::cerr << "FAILED: " << description << '\n';
      std::exit(1);
    }
  };
  auto ready = [&](const auto& n, const auto& p, const auto& v) {
    return joints_ready_at_home(n, p, v, home_names, home, 0.05, 1e-3);
  };
  require(ready(names, nominal, stationary), "loaded nominal start is valid");
  const std::vector<double> displaced = {
      -0.60313, -1.76431, -1.66483, -1.53366, 1.90767, 0.88341, 0.00368};
  require(!ready(names, displaced, stationary), "stationary displaced arm is invalid");
  auto reordered_names = names;
  auto reordered_positions = nominal;
  std::swap(reordered_names[0], reordered_names[5]);
  std::swap(reordered_positions[0], reordered_positions[5]);
  require(ready(reordered_names, reordered_positions, stationary), "match by name, not index");
  auto changed = nominal;
  changed[6] = 0.02;
  require(ready(names, changed, stationary), "no invented gripper home target");
  auto velocities = stationary;
  velocities[0] = 0.1;
  require(!ready(names, nominal, velocities), "moving arm is invalid");
  velocities = stationary;
  velocities[6] = 0.1;
  require(!ready(names, nominal, velocities), "moving gripper remains unsettled");
  velocities[6] = std::numeric_limits<double>::quiet_NaN();
  require(!ready(names, nominal, velocities), "NaN velocity is invalid");
  changed[1] = std::numeric_limits<double>::infinity();
  require(!ready(names, changed, stationary), "nonfinite position is invalid");
  changed = nominal;
  changed.pop_back();
  require(!ready(names, changed, stationary), "truncated positions are invalid");
  require(!ready(names, nominal, std::vector<double>{}), "missing velocities are invalid");
  reordered_names[0] = "unknown_joint";
  require(!ready(reordered_names, reordered_positions, stationary), "missing home joint is invalid");
  reordered_names = names;
  reordered_names[1] = names[0];
  require(!ready(reordered_names, nominal, stationary), "duplicate observed names are invalid");
  require(!joints_ready_at_home(names, nominal, stationary, {}, {}, 0.05, 1e-3),
          "empty home request is invalid");
  require(!joints_ready_at_home(names, nominal, stationary, home_names, home, -1.0, 1e-3),
          "negative tolerance is invalid");
  require(!joints_ready_at_home(names, nominal, stationary, home_names, home,
                               std::numeric_limits<double>::quiet_NaN(), 1e-3),
          "NaN tolerance is invalid");
  std::cout << checks << " joint-readiness checks passed\n";
}

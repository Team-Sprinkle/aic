/*
 * Copyright (C) 2026 Intrinsic Innovation LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef AIC_GAZEBO__JOINT_TARGET_PLUGIN_HH_
#define AIC_GAZEBO__JOINT_TARGET_PLUGIN_HH_

#include <gz/sim/EventManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/stringmsg.pb.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace aic_gazebo {
class JointTargetPlugin : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPreUpdate {
 public:
  void Configure(const gz::sim::Entity& _entity,
                 const std::shared_ptr<const sdf::Element>& _sdf,
                 gz::sim::EntityComponentManager& _ecm,
                 gz::sim::EventManager& _eventManager) override;

 public:
  void PreUpdate(const gz::sim::UpdateInfo& _info,
                 gz::sim::EntityComponentManager& _ecm) override;

 private:
  bool HandleJointTarget(const gz::msgs::StringMsg& _request,
                         gz::msgs::Boolean& _reply);

 private:
  gz::sim::Model model_;
  gz::transport::Node node_;
  std::string serviceName_{"/world/aic_world/joint_target"};
  std::string targetModelName_;
  std::unordered_map<std::string, double> requestedJoints_;
  std::unordered_map<std::string, double> activeJoints_;
  std::unordered_map<std::string, double> initialJoints_;
  int initialResetTicksRemaining_{0};
  std::unordered_map<std::string, gz::transport::Node::Publisher> publishers_;
  std::mutex mutex_;
};
}  // namespace aic_gazebo

#endif

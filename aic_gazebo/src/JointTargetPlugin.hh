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

#include <chrono>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace aic_gazebo {
class JointTargetPlugin : public gz::sim::System,
                          public gz::sim::ISystemConfigure,
                          public gz::sim::ISystemPreUpdate,
                          public gz::sim::ISystemReset {
 public:
  void Configure(const gz::sim::Entity& _entity,
                 const std::shared_ptr<const sdf::Element>& _sdf,
                 gz::sim::EntityComponentManager& _ecm,
                 gz::sim::EventManager& _eventManager) override;

 public:
  void PreUpdate(const gz::sim::UpdateInfo& _info,
                 gz::sim::EntityComponentManager& _ecm) override;

 public:
  void Reset(const gz::sim::UpdateInfo& _info,
             gz::sim::EntityComponentManager& _ecm) override;

 private:
  bool OnJointTargetRequest(const gz::msgs::StringMsg& _req,
                            gz::msgs::Boolean& _rep);

 private:
  bool ParseRequest(const std::string& _payload,
                    std::string& _modelName,
                    std::vector<std::string>& _jointNames,
                    std::vector<double>& _positions) const;

 private:
  gz::sim::Model model_;

 private:
  std::string modelName_;

 private:
  gz::transport::Node node_;

 private:
  std::mutex mutex_;

 private:
  std::unordered_map<std::string, double> requestedJointTargets_;

 private:
  std::chrono::steady_clock::duration updatePeriod_{0};

 private:
  std::chrono::steady_clock::duration lastUpdateTime_{0};

 private:
  std::string serviceName_;
};
}  // namespace aic_gazebo

#endif

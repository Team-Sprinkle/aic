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

#include "JointTargetPlugin.hh"

#include <gz/common/Console.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/JointPositionReset.hh>
#include <gz/sim/components/JointVelocityReset.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/World.hh>
#include <sstream>

GZ_ADD_PLUGIN(aic_gazebo::JointTargetPlugin, gz::sim::System,
              aic_gazebo::JointTargetPlugin::ISystemConfigure,
              aic_gazebo::JointTargetPlugin::ISystemPreUpdate,
              aic_gazebo::JointTargetPlugin::ISystemReset)

namespace aic_gazebo {
namespace {
std::vector<std::string> split(const std::string& input, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream stream(input);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    if (!item.empty()) {
      parts.push_back(item);
    }
  }
  return parts;
}
}  // namespace

void JointTargetPlugin::Configure(
    const gz::sim::Entity& _entity,
    const std::shared_ptr<const sdf::Element>& _sdf,
    gz::sim::EntityComponentManager& _ecm,
    gz::sim::EventManager& /*_eventManager*/) {
  double rate = _sdf->Get<double>("update_rate", 0).first;
  std::chrono::duration<double> period{rate > 0 ? 1 / rate : 0};
  this->updatePeriod_ =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);

  this->model_ = gz::sim::Model(_entity);
  this->modelName_ = this->model_.Name(_ecm);

  std::string worldName = "default";
  const auto parentEntity = _ecm.Component<gz::sim::components::ParentEntity>(_entity);
  if (parentEntity != nullptr) {
    const auto worldNameComp =
        _ecm.Component<gz::sim::components::Name>(parentEntity->Data());
    if (worldNameComp != nullptr && !worldNameComp->Data().empty()) {
      worldName = worldNameComp->Data();
    }
  }

  this->serviceName_ = std::string("/world/") + worldName + "/joint_target";
  if (!this->node_.Advertise(this->serviceName_,
                             &JointTargetPlugin::OnJointTargetRequest, this)) {
    gzerr << "Failed to advertise joint target service at " << this->serviceName_
          << std::endl;
    return;
  }
  gzmsg << "Initialized JointTargetPlugin service at " << this->serviceName_
        << std::endl;
}

void JointTargetPlugin::PreUpdate(const gz::sim::UpdateInfo& _info,
                                  gz::sim::EntityComponentManager& _ecm) {
  auto elapsed = _info.simTime - this->lastUpdateTime_;
  if (elapsed > std::chrono::steady_clock::duration::zero() &&
      elapsed < this->updatePeriod_) {
    return;
  }
  this->lastUpdateTime_ = _info.simTime;

  std::lock_guard<std::mutex> lock(this->mutex_);
  if (this->requestedJointTargets_.empty()) {
    return;
  }

  for (const auto& [jointName, jointTarget] : this->requestedJointTargets_) {
    auto jointEntity = this->model_.JointByName(_ecm, jointName);
    if (jointEntity == gz::sim::kNullEntity) {
      gzwarn << "JointTargetPlugin could not find joint " << jointName
             << std::endl;
      continue;
    }
    _ecm.SetComponentData<gz::sim::components::JointPositionReset>(
        jointEntity, {jointTarget});
    _ecm.SetComponentData<gz::sim::components::JointVelocityReset>(jointEntity,
                                                                   {0.0});
  }

  this->requestedJointTargets_.clear();
}

void JointTargetPlugin::Reset(const gz::sim::UpdateInfo& /*_info*/,
                              gz::sim::EntityComponentManager& /*_ecm*/) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  this->requestedJointTargets_.clear();
}

bool JointTargetPlugin::OnJointTargetRequest(const gz::msgs::StringMsg& _req,
                                             gz::msgs::Boolean& _rep) {
  std::string modelName;
  std::vector<std::string> jointNames;
  std::vector<double> positions;
  if (!this->ParseRequest(_req.data(), modelName, jointNames, positions)) {
    _rep.set_data(false);
    return true;
  }
  if (modelName != this->modelName_) {
    _rep.set_data(false);
    return true;
  }

  std::lock_guard<std::mutex> lock(this->mutex_);
  this->requestedJointTargets_.clear();
  for (std::size_t i = 0; i < jointNames.size(); ++i) {
    this->requestedJointTargets_[jointNames[i]] = positions[i];
  }
  _rep.set_data(true);
  return true;
}

bool JointTargetPlugin::ParseRequest(const std::string& _payload,
                                     std::string& _modelName,
                                     std::vector<std::string>& _jointNames,
                                     std::vector<double>& _positions) const {
  std::unordered_map<std::string, std::string> fields;
  for (const auto& part : split(_payload, ';')) {
    const auto delimiter = part.find('=');
    if (delimiter == std::string::npos) {
      continue;
    }
    fields[part.substr(0, delimiter)] = part.substr(delimiter + 1);
  }

  const auto modelIt = fields.find("model_name");
  const auto jointNamesIt = fields.find("joint_names");
  const auto positionsIt = fields.find("positions");
  if (modelIt == fields.end() || jointNamesIt == fields.end() ||
      positionsIt == fields.end()) {
    return false;
  }

  _modelName = modelIt->second;
  _jointNames = split(jointNamesIt->second, ',');
  const auto positionParts = split(positionsIt->second, ',');
  if (_jointNames.empty() || _jointNames.size() != positionParts.size()) {
    return false;
  }

  _positions.clear();
  _positions.reserve(positionParts.size());
  for (const auto& positionText : positionParts) {
    try {
      _positions.push_back(std::stod(positionText));
    } catch (...) {
      return false;
    }
  }
  return true;
}
}  // namespace aic_gazebo

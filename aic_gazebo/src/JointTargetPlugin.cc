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
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/components/JointPositionReset.hh>
#include <gz/sim/components/JointVelocityReset.hh>

#include <algorithm>
#include <sstream>
#include <vector>

GZ_ADD_PLUGIN(aic_gazebo::JointTargetPlugin, gz::sim::System,
              aic_gazebo::JointTargetPlugin::ISystemConfigure,
              aic_gazebo::JointTargetPlugin::ISystemPreUpdate)

namespace {
std::string Trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream stream(value);
  std::string part;
  while (std::getline(stream, part, delimiter)) {
    parts.push_back(Trim(part));
  }
  return parts;
}
}  // namespace

namespace aic_gazebo {
//////////////////////////////////////////////////
void JointTargetPlugin::Configure(
    const gz::sim::Entity& _entity,
    const std::shared_ptr<const sdf::Element>& _sdf,
    gz::sim::EntityComponentManager& /*_ecm*/,
    gz::sim::EventManager& /*_eventManager*/) {
  this->model_ = gz::sim::Model(_entity);
  std::string worldName = "aic_world";
  if (_sdf->HasElement("world_name")) {
    worldName = _sdf->Get<std::string>("world_name");
  }
  this->serviceName_ = "/world/" + worldName + "/joint_target";
  if (_sdf->HasElement("initial_joint_names") &&
      _sdf->HasElement("initial_positions")) {
    const auto jointNames = Split(_sdf->Get<std::string>("initial_joint_names"), ',');
    const auto positionTexts = Split(_sdf->Get<std::string>("initial_positions"), ',');
    if (jointNames.size() == positionTexts.size()) {
      for (std::size_t i = 0; i < jointNames.size(); ++i) {
        try {
          this->initialJoints_[jointNames[i]] = std::stod(positionTexts[i]);
        } catch (const std::exception&) {
          gzerr << "JointTargetPlugin initial position is not numeric: "
                << positionTexts[i] << std::endl;
        }
      }
    } else {
      gzerr << "JointTargetPlugin initial_joint_names and initial_positions "
            << "length mismatch." << std::endl;
    }
    this->initialResetTicksRemaining_ = _sdf->Get<int>("initial_reset_ticks", 30).first;
    if (this->initialResetTicksRemaining_ > 0) {
      this->activeJoints_ = this->initialJoints_;
    }
  }

  const bool advertised = this->node_.Advertise(
      this->serviceName_, &JointTargetPlugin::HandleJointTarget, this);
  if (!advertised) {
    gzerr << "Failed to advertise JointTargetPlugin service: "
          << this->serviceName_ << std::endl;
    return;
  }
  gzmsg << "Initialized JointTargetPlugin service: " << this->serviceName_
        << std::endl;
}

//////////////////////////////////////////////////
bool JointTargetPlugin::HandleJointTarget(const gz::msgs::StringMsg& _request,
                                          gz::msgs::Boolean& _reply) {
  std::unordered_map<std::string, std::string> fields;
  for (const auto& item : Split(_request.data(), ';')) {
    const auto idx = item.find('=');
    if (idx == std::string::npos) {
      continue;
    }
    fields[Trim(item.substr(0, idx))] = Trim(item.substr(idx + 1));
  }

  const auto modelIt = fields.find("model_name");
  const auto jointsIt = fields.find("joint_names");
  const auto positionsIt = fields.find("positions");
  if (modelIt == fields.end() || jointsIt == fields.end() ||
      positionsIt == fields.end()) {
    gzerr << "JointTargetPlugin request is missing model_name, joint_names, or positions: "
          << _request.data() << std::endl;
    _reply.set_data(false);
    return true;
  }

  const auto jointNames = Split(jointsIt->second, ',');
  const auto positionTexts = Split(positionsIt->second, ',');
  if (jointNames.empty() || jointNames.size() != positionTexts.size()) {
    gzerr << "JointTargetPlugin request has mismatched joint_names and positions: "
          << _request.data() << std::endl;
    _reply.set_data(false);
    return true;
  }

  std::unordered_map<std::string, double> requestedJoints;
  for (std::size_t i = 0; i < jointNames.size(); ++i) {
    if (jointNames[i].empty()) {
      _reply.set_data(false);
      return true;
    }
    try {
      requestedJoints[jointNames[i]] = std::stod(positionTexts[i]);
    } catch (const std::exception&) {
      gzerr << "JointTargetPlugin position is not numeric: "
            << positionTexts[i] << std::endl;
      _reply.set_data(false);
      return true;
    }
  }

  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->targetModelName_ = modelIt->second;
    this->requestedJoints_ = std::move(requestedJoints);
    this->activeJoints_ = this->requestedJoints_;
  }
  _reply.set_data(true);
  return true;
}

//////////////////////////////////////////////////
void JointTargetPlugin::PreUpdate(const gz::sim::UpdateInfo& /*_info*/,
                                  gz::sim::EntityComponentManager& _ecm) {
  std::unordered_map<std::string, double> requestedJoints;
  std::string targetModelName;
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (this->activeJoints_.empty() && this->initialResetTicksRemaining_ <= 0) {
      return;
    }
    targetModelName = this->targetModelName_;
    requestedJoints = this->activeJoints_.empty() ? this->initialJoints_ : this->activeJoints_;
    if (this->initialResetTicksRemaining_ > 0) {
      --this->initialResetTicksRemaining_;
    }
    this->requestedJoints_.clear();
  }

  const std::string modelName = this->model_.Name(_ecm);
  if (!targetModelName.empty() && targetModelName != modelName) {
    gzwarn << "JointTargetPlugin on model [" << modelName
           << "] received target for model [" << targetModelName
           << "]; applying to the plugin model." << std::endl;
  }

  for (const auto& [jointName, position] : requestedJoints) {
    const std::string topic = "/aic/joint_target/" + jointName;
    auto pubIt = this->publishers_.find(topic);
    if (pubIt == this->publishers_.end()) {
      auto publisher = this->node_.Advertise<gz::msgs::Double>(topic);
      pubIt = this->publishers_.emplace(topic, std::move(publisher)).first;
    }
    gz::msgs::Double command;
    command.set_data(position);
    pubIt->second.Publish(command);

    const auto jointEntity = this->model_.JointByName(_ecm, jointName);
    if (jointEntity == gz::sim::kNullEntity) {
      gzwarn << "JointTargetPlugin cannot find joint [" << jointName
             << "] on model [" << modelName << "]" << std::endl;
      continue;
    }
    _ecm.SetComponentData<gz::sim::components::JointPositionReset>(
        jointEntity, {position});
    _ecm.SetComponentData<gz::sim::components::JointVelocityReset>(
        jointEntity, {0.0});
  }
}
}  // namespace aic_gazebo

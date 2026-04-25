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

#include "WorldSdfGeneratorPlugin.hh"

#include <gz/msgs/sdf_generator_config.pb.h>
#include <gz/msgs/stringmsg.pb.h>

#include <fstream>
#include <gz/common/Console.hh>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/World.hh>

using namespace gz;
using namespace sim;

GZ_ADD_PLUGIN(aic_gazebo::WorldSdfGeneratorPlugin, gz::sim::System,
              aic_gazebo::WorldSdfGeneratorPlugin::ISystemConfigure,
              aic_gazebo::WorldSdfGeneratorPlugin::ISystemPostUpdate)

namespace {
inline constexpr char kLocalSaveWorldPath[] = "/tmp/aic.sdf";

std::string InjectOverviewCameraRigsIfMissing(const std::string &worldSdf) {
  if (worldSdf.find("overview_camera_rig") != std::string::npos) {
    return worldSdf;
  }
  const std::string closingWorldTag = "</world>";
  const std::size_t insertPos = worldSdf.rfind(closingWorldTag);
  if (insertPos == std::string::npos) {
    return worldSdf;
  }
  static const char *kOverviewRigModels = R"(
    <model name="overview_camera_rig">
      <static>true</static>
      <pose>0.95 -0.1 1.3 0.0 0.0 3.14</pose>
      <link name="overview_camera_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <pose>0 0 0 0 0 0</pose>
          <mass>0.001</mass>
          <inertia>
            <ixx>1e-6</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1e-6</iyy>
            <iyz>0</iyz>
            <izz>1e-6</izz>
          </inertia>
        </inertial>
        <visual name="overview_camera_visual">
          <geometry>
            <box>
              <size>0.02 0.02 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <sensor name="overview_camera" type="camera">
          <pose>0 0 0 0 0 0</pose>
          <frame_id>overview_camera_link</frame_id>
          <camera name="overview_camera">
            <pose>0 0 0 0 0 0</pose>
            <camera_info_topic>/overview_camera/camera_info</camera_info_topic>
            <optical_frame_id>overview_camera/optical</optical_frame_id>
            <image>
              <width>1152</width>
              <height>1024</height>
              <format>RGB_INT8</format>
              <anti_aliasing>4</anti_aliasing>
            </image>
            <horizontal_fov>0.8718</horizontal_fov>
            <clip>
              <near>0.07</near>
              <far>20</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>20</update_rate>
          <visualize>true</visualize>
          <topic>/overview_camera/image</topic>
          <enable_metrics>false</enable_metrics>
        </sensor>
      </link>
    </model>

    <model name="overview_front_camera_rig">
      <static>true</static>
      <pose>0.15 -1.05 1.22 1.5708 0.0 1.5708</pose>
      <link name="overview_front_camera_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <pose>0 0 0 0 0 0</pose>
          <mass>0.001</mass>
          <inertia>
            <ixx>1e-6</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1e-6</iyy>
            <iyz>0</iyz>
            <izz>1e-6</izz>
          </inertia>
        </inertial>
        <visual name="overview_front_camera_visual">
          <geometry>
            <box>
              <size>0.02 0.02 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <sensor name="overview_front_camera" type="camera">
          <pose>0 0 0 0 0 0</pose>
          <frame_id>overview_front_camera_link</frame_id>
          <camera name="overview_front_camera">
            <pose>0 0 0 0 0 0</pose>
            <camera_info_topic>/overview_front_camera/camera_info</camera_info_topic>
            <optical_frame_id>overview_front_camera/optical</optical_frame_id>
            <image>
              <width>1152</width>
              <height>1024</height>
              <format>RGB_INT8</format>
              <anti_aliasing>4</anti_aliasing>
            </image>
            <horizontal_fov>0.8718</horizontal_fov>
            <clip>
              <near>0.07</near>
              <far>20</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>20</update_rate>
          <visualize>true</visualize>
          <topic>/overview_front_camera/image</topic>
          <enable_metrics>false</enable_metrics>
        </sensor>
      </link>
    </model>

    <model name="overview_side_camera_rig">
      <static>true</static>
      <pose>-0.95 -0.15 1.18 0.0 0.0 0.0</pose>
      <link name="overview_side_camera_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <pose>0 0 0 0 0 0</pose>
          <mass>0.001</mass>
          <inertia>
            <ixx>1e-6</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1e-6</iyy>
            <iyz>0</iyz>
            <izz>1e-6</izz>
          </inertia>
        </inertial>
        <visual name="overview_side_camera_visual">
          <geometry>
            <box>
              <size>0.02 0.02 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <sensor name="overview_side_camera" type="camera">
          <pose>0 0 0 0 0 0</pose>
          <frame_id>overview_side_camera_link</frame_id>
          <camera name="overview_side_camera">
            <pose>0 0 0 0 0 0</pose>
            <camera_info_topic>/overview_side_camera/camera_info</camera_info_topic>
            <optical_frame_id>overview_side_camera/optical</optical_frame_id>
            <image>
              <width>1152</width>
              <height>1024</height>
              <format>RGB_INT8</format>
              <anti_aliasing>4</anti_aliasing>
            </image>
            <horizontal_fov>0.8718</horizontal_fov>
            <clip>
              <near>0.07</near>
              <far>20</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>20</update_rate>
          <visualize>true</visualize>
          <topic>/overview_side_camera/image</topic>
          <enable_metrics>false</enable_metrics>
        </sensor>
      </link>
    </model>

    <model name="overview_oblique_camera_rig">
      <static>true</static>
      <pose>0.95 -0.95 1.55 0.0 0.0 2.35</pose>
      <link name="overview_oblique_camera_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial>
          <pose>0 0 0 0 0 0</pose>
          <mass>0.001</mass>
          <inertia>
            <ixx>1e-6</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1e-6</iyy>
            <iyz>0</iyz>
            <izz>1e-6</izz>
          </inertia>
        </inertial>
        <visual name="overview_oblique_camera_visual">
          <geometry>
            <box>
              <size>0.02 0.02 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.2 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <sensor name="overview_oblique_camera" type="camera">
          <pose>0 0 0 0 0 0</pose>
          <frame_id>overview_oblique_camera_link</frame_id>
          <camera name="overview_oblique_camera">
            <pose>0 0 0 0 0 0</pose>
            <camera_info_topic>/overview_oblique_camera/camera_info</camera_info_topic>
            <optical_frame_id>overview_oblique_camera/optical</optical_frame_id>
            <image>
              <width>1152</width>
              <height>1024</height>
              <format>RGB_INT8</format>
              <anti_aliasing>4</anti_aliasing>
            </image>
            <horizontal_fov>0.8718</horizontal_fov>
            <clip>
              <near>0.07</near>
              <far>20</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>20</update_rate>
          <visualize>true</visualize>
          <topic>/overview_oblique_camera/image</topic>
          <enable_metrics>false</enable_metrics>
        </sensor>
      </link>
    </model>
)";
  std::string patched = worldSdf;
  patched.insert(insertPos, kOverviewRigModels);
  return patched;
}
}

namespace aic_gazebo {

//////////////////////////////////////////////////
void WorldSdfGeneratorPlugin::Configure(
    const gz::sim::Entity&, const std::shared_ptr<const sdf::Element>& _sdf,
    gz::sim::EntityComponentManager&, gz::sim::EventManager&) {
  gzdbg << "aic_gazebo::WorldSdfGeneratorPlugin::Configure" << std::endl;

  double delay = _sdf->Get<double>("save_world_delay_s", 0.0).first;
  this->saveWorldDelay = std::chrono::duration<double>(delay);
  const double forceDelay =
      _sdf->Get<double>("force_save_after_s", delay + 30.0).first;
  this->forceSaveAfter = std::chrono::duration<double>(forceDelay);
  this->saveWorldPath =
      _sdf->Get<std::string>("save_world_path", kLocalSaveWorldPath).first;

  this->requiredEntityNames = {"task_board", "cable_0"};
}

//////////////////////////////////////////////////
void WorldSdfGeneratorPlugin::PostUpdate(
    const gz::sim::UpdateInfo& _info,
    const gz::sim::EntityComponentManager& _ecm) {
  if (this->sdfGenerated) return;

  // Wait for specified delay duration before requesting to save world
  if (_info.simTime < this->saveWorldDelay) return;

  if (!this->requiredEntityNames.empty()) {
    std::vector<std::string> missingNames;
    for (const auto &requiredName : this->requiredEntityNames) {
      if (_ecm.EntityByComponents(components::Name(requiredName)) == kNullEntity) {
        missingNames.push_back(requiredName);
      }
    }
    if (!missingNames.empty() && _info.simTime < this->forceSaveAfter) {
      gzdbg << "Delaying world export; waiting for entities:";
      for (const auto &name : missingNames) {
        gzdbg << " " << name;
      }
      gzdbg << std::endl;
      return;
    }
    if (!missingNames.empty()) {
      gzmsg << "Forcing world export without all required entities after "
            << this->forceSaveAfter.count() << "s. Missing:";
      for (const auto &name : missingNames) {
        gzmsg << " " << name;
      }
      gzmsg << std::endl;
    }
  }

  Entity world = _ecm.EntityByComponents(components::World());
  auto nameComp = _ecm.Component<components::Name>(world);
  std::string worldName = nameComp->Data();
  const std::string sdfGenService{std::string("/world/") + worldName +
                                  "/generate_world_sdf"};
  msgs::StringMsg genWorldSdf;
  msgs::SdfGeneratorConfig req;
  auto* globalConfig = req.mutable_global_entity_gen_config();
  msgs::Set(globalConfig->mutable_expand_include_tags(), true);

  const unsigned int timeout{5000};
  bool result = false;
  bool serviceCall =
      this->node.Request(sdfGenService, req, timeout, genWorldSdf, result);
  if (serviceCall && result && !genWorldSdf.data().empty()) {
    const std::string worldSdf = InjectOverviewCameraRigsIfMissing(genWorldSdf.data());
    gzdbg << "Saving world: " << worldName << " to: " << this->saveWorldPath
          << std::endl;
    std::ofstream fs(this->saveWorldPath, std::ios::out);
    if (fs.is_open()) {
      fs << worldSdf;
      gzmsg << "World saved to " << this->saveWorldPath << std::endl;
    } else {
      gzmsg << "File: " << this->saveWorldPath << " could not be opened for "
            << "saving. Please check that the directory containg the "
            << "file exists and the correct permissions are set." << std::endl;
    }
  } else {
    if (!serviceCall) {
      gzmsg << "Service call for generating world SDF timed out" << std::endl;
    }
    gzmsg << "Unknown error occured when saving the world." << std::endl;
  }

  this->sdfGenerated = true;
}
}  // namespace aic_gazebo

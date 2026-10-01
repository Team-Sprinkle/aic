// Training-label instrumentation only. Follow existing wrist-camera link poses
// with visual-only world cameras. Never change a robot pose or collision.
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Visibility.hh>
#include <gz/sim/components/Visual.hh>
#include <array>
#include <string>

namespace aic_gazebo {
class LabelCameraFollowerPlugin : public gz::sim::System,
    public gz::sim::ISystemPreUpdate {
 public:
  void PreUpdate(const gz::sim::UpdateInfo &,
                 gz::sim::EntityComponentManager &ecm) override {
    // Ordinary RGB retains all these visuals. Label cameras use bit 2, and
    // ignore only camera-body meshes normally clipped out of their self-view.
    // Gripper, arm, cable and cards remain visible to the label cameras.
    ecm.Each<gz::sim::components::Visual, gz::sim::components::Name>(
        [&](const gz::sim::Entity &entity, const gz::sim::components::Visual *,
            const gz::sim::components::Name *name) {
          if (name->Data().find("basler_cam_visual") == std::string::npos &&
              name->Data().find("camera_link_visual") == std::string::npos) return true;
          auto *flags = ecm.Component<gz::sim::components::VisibilityFlags>(entity);
          if (!flags) ecm.CreateComponent(entity, gz::sim::components::VisibilityFlags(1u));
          else if (flags->SetData(1u, [](auto a, auto b) { return a == b; }))
            ecm.SetChanged(entity, gz::sim::components::VisibilityFlags::typeId,
                           gz::sim::ComponentState::OneTimeChange);
          return true;
        });
    for (const auto &camera : std::array<std::string, 3>{"center", "left", "right"}) {
      gz::sim::Entity source = gz::sim::kNullEntity;
      gz::sim::Entity target = gz::sim::kNullEntity;
      ecm.Each<gz::sim::components::Link, gz::sim::components::Name>(
          [&](const gz::sim::Entity &entity, const gz::sim::components::Link *,
              const gz::sim::components::Name *name) {
            if (name->Data() == camera + "_camera/sensor_link") source = entity;
            return true;
          });
      ecm.Each<gz::sim::components::Model, gz::sim::components::Name>(
          [&](const gz::sim::Entity &entity, const gz::sim::components::Model *,
              const gz::sim::components::Name *name) {
            if (name->Data() == "label_only_" + camera) target = entity;
            return true;
          });
      if (source == gz::sim::kNullEntity || target == gz::sim::kNullEntity) continue;
      auto *pose = ecm.Component<gz::sim::components::Pose>(target);
      if (!pose) continue;
      const auto measured = gz::sim::worldPose(source, ecm);
      if (pose->SetData(measured, [](const auto &a, const auto &b) { return a == b; }))
        ecm.SetChanged(target, gz::sim::components::Pose::typeId,
                       gz::sim::ComponentState::OneTimeChange);
    }
  }
};
}
GZ_ADD_PLUGIN(aic_gazebo::LabelCameraFollowerPlugin, gz::sim::System,
             aic_gazebo::LabelCameraFollowerPlugin::ISystemPreUpdate)

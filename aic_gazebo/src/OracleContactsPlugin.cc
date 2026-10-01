// Diagnostic only: request contact reporting for every collision and publish
// complete, named contact pairs. No geometry, friction, or masks are changed.
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/transport/Node.hh>
#include <set>

namespace aic_gazebo {
class OracleContactsPlugin : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
    public gz::sim::ISystemPostUpdate {
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    publisher = node.Advertise<gz::msgs::Contacts>("/audit/oracle/contacts");
    correctedPublisher = node.Advertise<gz::msgs::Contacts>("/audit/oracle/contacts_corrected");
    legacyImpulse = sdf->Get<bool>("legacy_bullet_impulse_contacts", false).first;
  }
  void PreUpdate(const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &ecm) override {
    ecm.Each<gz::sim::components::Collision>([&](const gz::sim::Entity &entity,
        const gz::sim::components::Collision *) {
      if (!ecm.Component<gz::sim::components::ContactSensorData>(entity))
        ecm.CreateComponent(entity, gz::sim::components::ContactSensorData());
      return true;
    });
  }
  void PostUpdate(const gz::sim::UpdateInfo &info,
      const gz::sim::EntityComponentManager &ecm) override {
    if (info.paused || (info.simTime >= lastTime &&
        info.simTime - lastTime < std::chrono::milliseconds(10))) return;
    lastTime = info.simTime;
    gz::msgs::Contacts out;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
    out.mutable_header()->mutable_stamp()->set_sec(ns / 1000000000);
    out.mutable_header()->mutable_stamp()->set_nsec(ns % 1000000000);
    std::set<std::pair<uint64_t, uint64_t>> seen;
    ecm.Each<gz::sim::components::ContactSensorData>([&](const gz::sim::Entity &,
        const gz::sim::components::ContactSensorData *data) {
      for (const auto &contact : data->Data().contact()) {
        const auto a = contact.collision1().id(), b = contact.collision2().id();
        if (!seen.insert(std::minmax(a, b)).second) continue;
        auto *copy = out.add_contact();
        *copy = contact;
        copy->mutable_collision1()->set_name(gz::sim::scopedName(a, ecm, "::", false));
        copy->mutable_collision2()->set_name(gz::sim::scopedName(b, ecm, "::", false));
      }
      return true;
    });
    publisher.Publish(out);  // Raw messages are always preserved.
    if (legacyImpulse && info.dt.count() > 0) {
      // Backport only reporting arithmetic from gz-physics #890. No solver changes.
      const double dt = std::chrono::duration<double>(info.dt).count();
      for (auto &contact : *out.mutable_contact()) {
        for (int i = 0; i < contact.wrench_size() && i < contact.normal_size(); ++i) {
          const double force = contact.wrench(i).body_1_wrench().force().x() / dt;
          const auto &normal = contact.normal(i);
          auto *a = contact.mutable_wrench(i)->mutable_body_1_wrench()->mutable_force();
          auto *b = contact.mutable_wrench(i)->mutable_body_2_wrench()->mutable_force();
          a->set_x(force * normal.x()); a->set_y(force * normal.y()); a->set_z(force * normal.z());
          b->set_x(-a->x()); b->set_y(-a->y()); b->set_z(-a->z());
        }
        for (int i = 0; i < contact.depth_size(); ++i) contact.set_depth(i, -contact.depth(i));
      }
      correctedPublisher.Publish(out);
    }
  }
 private:
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher, correctedPublisher;
  bool legacyImpulse{false};
  std::chrono::steady_clock::duration lastTime{0};
};
}
GZ_ADD_PLUGIN(aic_gazebo::OracleContactsPlugin, gz::sim::System,
    aic_gazebo::OracleContactsPlugin::ISystemConfigure,
    aic_gazebo::OracleContactsPlugin::ISystemPreUpdate,
    aic_gazebo::OracleContactsPlugin::ISystemPostUpdate)

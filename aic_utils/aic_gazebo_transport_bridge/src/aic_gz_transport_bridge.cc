#include <google/protobuf/message.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/serialized_map.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/transport/Node.hh>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <chrono>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

struct TopicSample {
  std::string text;
  std::size_t generation{0};
  Clock::time_point received_at{};
};

class BridgeServer {
 public:
  BridgeServer(std::string stateTopic, std::string poseTopic)
      : stateTopic_(std::move(stateTopic)), poseTopic_(std::move(poseTopic)) {}

  bool Start(std::string &error) {
    gz::transport::RawCallback stateCb =
        [this](const char *_msgData,
               std::size_t _size,
               const gz::transport::MessageInfo &) {
          this->OnStateRaw(_msgData, _size);
        };
    if (!this->node_.SubscribeRaw(this->stateTopic_, stateCb)) {
      error = "failed to subscribe to state topic " + this->stateTopic_;
      return false;
    }

    gz::transport::RawCallback poseCb =
        [this](const char *_msgData,
               std::size_t _size,
               const gz::transport::MessageInfo &) {
          this->OnPoseRaw(_msgData, _size);
        };
    if (!this->node_.SubscribeRaw(this->poseTopic_, poseCb)) {
      error = "failed to subscribe to pose topic " + this->poseTopic_;
      return false;
    }
    return true;
  }

  rapidjson::Document Handle(const rapidjson::Document &request) {
    rapidjson::Document response;
    response.SetObject();
    auto &allocator = response.GetAllocator();
    if (!request.HasMember("id")) {
      this->AddError(response, "missing request id");
      return response;
    }
    if (request["id"].IsUint64()) {
      response.AddMember("id", request["id"].GetUint64(), allocator);
    } else if (request["id"].IsInt64()) {
      response.AddMember("id", request["id"].GetInt64(), allocator);
    } else {
      this->AddError(response, "request id must be an integer");
      return response;
    }
    if (!request.HasMember("op") || !request["op"].IsString()) {
      this->AddError(response, "missing string op");
      return response;
    }

    const std::string op = request["op"].GetString();
    if (op == "ping") {
      response.AddMember("ok", true, allocator);
      return response;
    }
    if (op == "shutdown") {
      response.AddMember("ok", true, allocator);
      this->shutdownRequested_ = true;
      return response;
    }
    if (op == "get_observation") {
      return this->HandleGetObservation(request, std::move(response));
    }
    if (op == "status") {
      return this->HandleStatus(std::move(response));
    }
    if (op == "wait_until_ready") {
      return this->HandleWaitUntilReady(request, std::move(response));
    }
    if (op == "world_control") {
      return this->HandleWorldControl(request, std::move(response));
    }
    if (op == "set_pose") {
      return this->HandleSetPose(request, std::move(response));
    }
    if (op == "joint_target") {
      return this->HandleJointTarget(request, std::move(response));
    }
    this->AddError(response, "unsupported op: " + op);
    return response;
  }

  bool ShutdownRequested() const { return this->shutdownRequested_; }

 private:
  rapidjson::Document HandleGetObservation(const rapidjson::Document &request,
                                           rapidjson::Document response) {
    auto &allocator = response.GetAllocator();
    const std::size_t afterGeneration =
        this->ReadOptionalUint(request, "after_generation").value_or(0);
    const auto timeoutMs =
        this->ReadOptionalInt(request, "timeout_ms").value_or(1000);
    const auto poseTimeoutMs =
        this->ReadOptionalInt(request, "pose_timeout_ms").value_or(200);

    auto state = this->WaitForState(afterGeneration, Milliseconds(timeoutMs));
    if (!state.has_value()) {
      this->AddError(response, "timed out waiting for state sample");
      return response;
    }

    auto pose = this->LatestPose(Milliseconds(poseTimeoutMs));

    response.AddMember("ok", true, allocator);
    response.AddMember(
        "state_generation",
        static_cast<uint64_t>(state->generation),
        allocator);
    response.AddMember(
        "state_text",
        rapidjson::Value(state->text.c_str(), allocator),
        allocator);
    if (pose.has_value()) {
      response.AddMember(
          "pose_generation",
          static_cast<uint64_t>(pose->generation),
          allocator);
      response.AddMember(
          "pose_text",
          rapidjson::Value(pose->text.c_str(), allocator),
          allocator);
    } else {
      response.AddMember("pose_generation", rapidjson::Value().SetNull(), allocator);
      response.AddMember("pose_text", rapidjson::Value().SetNull(), allocator);
    }
    return response;
  }

  rapidjson::Document HandleStatus(rapidjson::Document response) {
    return this->PopulateStatus(std::move(response), true);
  }

  rapidjson::Document HandleWaitUntilReady(const rapidjson::Document &request,
                                           rapidjson::Document response) {
    const auto timeoutMs =
        this->ReadOptionalInt(request, "timeout_ms").value_or(1000);
    const bool requirePose =
        !request.HasMember("require_pose") || request["require_pose"].GetBool();
    const bool ready =
        this->WaitUntilReady(Milliseconds(timeoutMs), requirePose);
    response = this->PopulateStatus(std::move(response), ready);
    if (!ready) {
      this->AddError(response, "timed out waiting for initial transport samples");
    }
    return response;
  }

  rapidjson::Document PopulateStatus(rapidjson::Document response, bool ok) {
    auto &allocator = response.GetAllocator();
    std::lock_guard<std::mutex> lock(this->mutex_);
    response.AddMember("ok", ok, allocator);
    response.AddMember(
        "state_generation",
        static_cast<uint64_t>(this->state_.generation),
        allocator);
    response.AddMember(
        "pose_generation",
        static_cast<uint64_t>(this->pose_.generation),
        allocator);
    response.AddMember(
        "state_callback_count",
        static_cast<uint64_t>(this->stateCallbackCount_),
        allocator);
    response.AddMember(
        "state_parse_failures",
        static_cast<uint64_t>(this->stateParseFailureCount_),
        allocator);
    response.AddMember(
        "pose_callback_count",
        static_cast<uint64_t>(this->poseCallbackCount_),
        allocator);
    response.AddMember(
        "pose_parse_failures",
        static_cast<uint64_t>(this->poseParseFailureCount_),
        allocator);
    return response;
  }

  rapidjson::Document HandleWorldControl(const rapidjson::Document &request,
                                         rapidjson::Document response) {
    auto &allocator = response.GetAllocator();
    if (!request.HasMember("service") || !request["service"].IsString()) {
      this->AddError(response, "world_control requires string service");
      return response;
    }
    gz::msgs::WorldControl req;
    if (request.HasMember("multi_step")) {
      if (!request["multi_step"].IsUint()) {
        this->AddError(response, "world_control.multi_step must be uint");
        return response;
      }
      req.set_multi_step(request["multi_step"].GetUint());
    }
    if (request.HasMember("reset_all")) {
      if (!request["reset_all"].IsBool()) {
        this->AddError(response, "world_control.reset_all must be bool");
        return response;
      }
      req.mutable_reset()->set_all(request["reset_all"].GetBool());
    }
    const unsigned int timeoutMs =
        static_cast<unsigned int>(this->ReadOptionalInt(request, "timeout_ms").value_or(1000));
    gz::msgs::Boolean rep;
    bool result = false;
    const bool requested = this->RequestWithRetry(
        request["service"].GetString(), req, timeoutMs, rep, result);
    response.AddMember("ok", requested && result && rep.data(), allocator);
    response.AddMember("requested", requested, allocator);
    response.AddMember("result", result, allocator);
    response.AddMember("reply_text", rapidjson::Value(rep.DebugString().c_str(), allocator), allocator);
    if (!(requested && result && rep.data())) {
      this->AddError(response, "world control request failed");
    }
    return response;
  }

  rapidjson::Document HandleSetPose(const rapidjson::Document &request,
                                    rapidjson::Document response) {
    auto &allocator = response.GetAllocator();
    if (!request.HasMember("service") || !request["service"].IsString()) {
      this->AddError(response, "set_pose requires string service");
      return response;
    }
    if (!request.HasMember("name") || !request["name"].IsString()) {
      this->AddError(response, "set_pose requires string name");
      return response;
    }
    const auto position = this->ReadVector(request, "position", 3);
    const auto orientation = this->ReadVector(request, "orientation", 4);
    if (!position.has_value() || !orientation.has_value()) {
      this->AddError(response, "set_pose requires numeric position[3] and orientation[4]");
      return response;
    }

    gz::msgs::Pose req;
    req.set_name(request["name"].GetString());
    req.mutable_position()->set_x((*position)[0]);
    req.mutable_position()->set_y((*position)[1]);
    req.mutable_position()->set_z((*position)[2]);
    req.mutable_orientation()->set_x((*orientation)[0]);
    req.mutable_orientation()->set_y((*orientation)[1]);
    req.mutable_orientation()->set_z((*orientation)[2]);
    req.mutable_orientation()->set_w((*orientation)[3]);

    const unsigned int timeoutMs =
        static_cast<unsigned int>(this->ReadOptionalInt(request, "timeout_ms").value_or(1000));
    gz::msgs::Boolean rep;
    bool result = false;
    const bool requested = this->RequestWithRetry(
        request["service"].GetString(), req, timeoutMs, rep, result);
    response.AddMember("ok", requested && result && rep.data(), allocator);
    response.AddMember("requested", requested, allocator);
    response.AddMember("result", result, allocator);
    response.AddMember("reply_text", rapidjson::Value(rep.DebugString().c_str(), allocator), allocator);
    if (!(requested && result && rep.data())) {
      this->AddError(response, "set_pose request failed");
    }
    return response;
  }

  rapidjson::Document HandleJointTarget(const rapidjson::Document &request,
                                        rapidjson::Document response) {
    auto &allocator = response.GetAllocator();
    if (!request.HasMember("service") || !request["service"].IsString()) {
      this->AddError(response, "joint_target requires string service");
      return response;
    }
    if (!request.HasMember("request_text") || !request["request_text"].IsString()) {
      this->AddError(response, "joint_target requires string request_text");
      return response;
    }

    gz::msgs::StringMsg req;
    req.set_data(request["request_text"].GetString());

    const unsigned int timeoutMs =
        static_cast<unsigned int>(this->ReadOptionalInt(request, "timeout_ms").value_or(1000));
    gz::msgs::Boolean rep;
    bool result = false;
    const bool requested = this->RequestWithRetry(
        request["service"].GetString(), req, timeoutMs, rep, result);
    response.AddMember("ok", requested && result && rep.data(), allocator);
    response.AddMember("requested", requested, allocator);
    response.AddMember("result", result, allocator);
    response.AddMember("reply_text", rapidjson::Value(rep.DebugString().c_str(), allocator), allocator);
    if (!(requested && result && rep.data())) {
      this->AddError(response, "joint_target request failed");
    }
    return response;
  }

  void AddError(rapidjson::Document &response, const std::string &message) const {
    auto &allocator = response.GetAllocator();
    if (!response.HasMember("ok")) {
      response.AddMember("ok", false, allocator);
    } else {
      response["ok"].SetBool(false);
    }
    if (response.HasMember("error")) {
      response["error"].SetString(message.c_str(), allocator);
    } else {
      response.AddMember("error", rapidjson::Value(message.c_str(), allocator), allocator);
    }
  }

  void OnStateRaw(const char *msgData, std::size_t size) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->stateCallbackCount_ += 1;
    gz::msgs::SerializedStepMap msg;
    if (!msg.ParseFromArray(msgData, static_cast<int>(size))) {
      this->stateParseFailureCount_ += 1;
      return;
    }
    this->state_.text = msg.DebugString();
    this->state_.generation += 1;
    this->state_.received_at = Clock::now();
    this->condition_.notify_all();
  }

  void OnPoseRaw(const char *msgData, std::size_t size) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->poseCallbackCount_ += 1;
    gz::msgs::Pose_V msg;
    if (!msg.ParseFromArray(msgData, static_cast<int>(size))) {
      this->poseParseFailureCount_ += 1;
      return;
    }
    this->pose_.text = msg.DebugString();
    this->pose_.generation += 1;
    this->pose_.received_at = Clock::now();
    this->condition_.notify_all();
  }

  std::optional<TopicSample> WaitForState(std::size_t afterGeneration,
                                          Milliseconds timeout) {
    const auto deadline = Clock::now() + timeout;
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->condition_.wait_until(lock, deadline, [this, afterGeneration]() {
      return this->state_.generation > afterGeneration && !this->state_.text.empty();
    });
    if (this->state_.generation <= afterGeneration || this->state_.text.empty()) {
      return std::nullopt;
    }
    return this->state_;
  }

  bool WaitUntilReady(Milliseconds timeout, bool requirePose) {
    const auto deadline = Clock::now() + timeout;
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->condition_.wait_until(lock, deadline, [this, requirePose]() {
      const bool stateReady =
          this->state_.generation > 0 && !this->state_.text.empty() &&
          this->stateCallbackCount_ > 0;
      const bool poseReady =
          !requirePose ||
          ((this->pose_.generation > 0 && !this->pose_.text.empty()) ||
           this->poseCallbackCount_ > 0);
      return stateReady && poseReady;
    });
    const bool stateReady =
        this->state_.generation > 0 && !this->state_.text.empty() &&
        this->stateCallbackCount_ > 0;
    const bool poseReady =
        !requirePose ||
        ((this->pose_.generation > 0 && !this->pose_.text.empty()) ||
         this->poseCallbackCount_ > 0);
    return stateReady && poseReady;
  }

  std::optional<TopicSample> LatestPose(Milliseconds maxAge) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (this->pose_.text.empty()) {
      return std::nullopt;
    }
    if (maxAge.count() > 0 && this->pose_.generation == 0) {
      return std::nullopt;
    }
    if (maxAge.count() > 0 &&
        Clock::now() - this->pose_.received_at > maxAge) {
      return this->pose_;
    }
    return this->pose_;
  }

  std::optional<int> ReadOptionalInt(const rapidjson::Document &request,
                                     const char *key) const {
    if (!request.HasMember(key)) {
      return std::nullopt;
    }
    if (!request[key].IsInt()) {
      return std::nullopt;
    }
    return request[key].GetInt();
  }

  std::optional<std::size_t> ReadOptionalUint(const rapidjson::Document &request,
                                              const char *key) const {
    if (!request.HasMember(key)) {
      return std::nullopt;
    }
    if (request[key].IsUint64()) {
      return static_cast<std::size_t>(request[key].GetUint64());
    }
    if (request[key].IsUint()) {
      return static_cast<std::size_t>(request[key].GetUint());
    }
    return std::nullopt;
  }

  std::optional<std::array<double, 4>> ReadVector(const rapidjson::Document &request,
                                                  const char *key,
                                                  std::size_t expectedSize) const {
    if (!request.HasMember(key) || !request[key].IsArray() ||
        request[key].Size() != expectedSize) {
      return std::nullopt;
    }
    std::array<double, 4> values{0.0, 0.0, 0.0, 0.0};
    for (rapidjson::SizeType i = 0; i < request[key].Size(); ++i) {
      if (!request[key][i].IsNumber()) {
        return std::nullopt;
      }
      values[i] = request[key][i].GetDouble();
    }
    return values;
  }

  template <typename RequestT, typename ReplyT>
  bool RequestWithRetry(const std::string &service,
                        const RequestT &request,
                        unsigned int timeoutMs,
                        ReplyT &reply,
                        bool &result) {
    constexpr int maxAttempts = 4;
    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
      const bool requested =
          this->node_.Request(service, request, timeoutMs, reply, result);
      if (requested) {
        return true;
      }
      if (attempt + 1 < maxAttempts) {
        std::this_thread::sleep_for(Milliseconds(250));
      }
    }
    result = false;
    return false;
  }

  gz::transport::Node node_;
  std::string stateTopic_;
  std::string poseTopic_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  TopicSample state_;
  TopicSample pose_;
  std::size_t stateCallbackCount_{0};
  std::size_t stateParseFailureCount_{0};
  std::size_t poseCallbackCount_{0};
  std::size_t poseParseFailureCount_{0};
  bool shutdownRequested_{false};
};

std::string ToJsonString(const rapidjson::Document &document) {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  document.Accept(writer);
  return buffer.GetString();
}

}  // namespace

int main(int argc, char **argv) {
  std::string stateTopic;
  std::string poseTopic;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--state-topic") == 0 && i + 1 < argc) {
      stateTopic = argv[++i];
      continue;
    }
    if (std::strcmp(argv[i], "--pose-topic") == 0 && i + 1 < argc) {
      poseTopic = argv[++i];
      continue;
    }
  }

  if (stateTopic.empty() || poseTopic.empty()) {
    std::cerr << "usage: aic_gz_transport_bridge --state-topic <topic> --pose-topic <topic>"
              << std::endl;
    return 2;
  }

  BridgeServer server(stateTopic, poseTopic);
  std::string error;
  if (!server.Start(error)) {
    std::cerr << error << std::endl;
    return 1;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    rapidjson::Document request;
    request.Parse(line.c_str());
    rapidjson::Document response;
    if (request.HasParseError() || !request.IsObject()) {
      response.SetObject();
      auto &allocator = response.GetAllocator();
      response.AddMember("ok", false, allocator);
      response.AddMember(
          "error",
          rapidjson::Value("invalid json request", allocator),
          allocator);
    } else {
      response = server.Handle(request);
    }
    std::cout << ToJsonString(response) << std::endl;
    if (server.ShutdownRequested()) {
      break;
    }
  }
  return 0;
}

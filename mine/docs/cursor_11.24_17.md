# 综合代码优化建议
_Exported on 2025/11/24 at GMT+8 17:47:08 from Cursor (2.1.20)_

---

**User**

@QR/docs/cursor_claude.md @QR/docs/cursor_composer.md @QR/docs/cursor_gemini.md @QR/docs/cursor_gpt.md 综合一下，给我最优的代码

---

**Cursor**

```cpp
#ifndef ORBSLAM3_EXPORTER_H
#define ORBSLAM3_EXPORTER_H

#include <cstddef>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "Camera.h"

/**
 * ORBSLAM3Exporter funnels front-end data (keyframe observations, IMU segments,
 * priors) into the VariableEliminationDriver, keeping bookkeeping about which
 * variables still need to be eliminated.
 */
class ORBSLAM3Exporter {
public:
    struct ObservationInput {
        int pose_id = -1;
        int landmark_id = -1;
        Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        Eigen::Vector3f landmark_w = Eigen::Vector3f::Zero();
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        float sigma_pixel = 1.0f;
    };

    struct ImuEdgeInput {
        int pose_i_id = -1;
        int pose_j_id = -1;
        std::vector<ImuPreprocessorGTSAM::ImuMeasurement> measurements;
        double dt_total = 0.0;
    };

    struct PriorInput {
        int pose_id = -1;
        Eigen::Matrix3f Rp = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tp = Eigen::Vector3f::Zero();
        Eigen::Matrix<float,6,1> sigma = Eigen::Matrix<float,6,1>::Constant(1.0f);
    };

    explicit ORBSLAM3Exporter(VariableEliminationDriver& driver);

    void addObservation(const ObservationInput& obs_input);
    void addObservations(const std::vector<ObservationInput>& obs_inputs);

    void addImuEdge(const ImuEdgeInput& edge_input);
    void addImuEdges(const std::vector<ImuEdgeInput>& edge_inputs);

    void addPrior(const PriorInput& prior_input);

    bool eliminateLandmark(int landmark_id, float huber_delta = 2.5f);
    std::size_t eliminateLandmarks(const std::vector<int>& landmark_ids, float huber_delta = 2.5f);
    std::size_t eliminateAllLandmarks(float huber_delta = 2.5f);

    bool eliminatePose(int pose_id);
    std::size_t eliminatePoses(const std::vector<int>& pose_ids);

    std::vector<int> pendingLandmarkIds() const;
    std::vector<int> pendingPoseIds() const;

    void clear();

private:
    struct ImuConstraintEntry {
        ImuConstraint constraint;
        bool used = false;
    };

    VariableEliminationDriver& driver_;
    std::unordered_map<int, std::vector<CameraObservation>> landmark_observations_;
    std::vector<ImuConstraintEntry> imu_constraints_;
    std::unordered_map<int, std::vector<std::size_t>> pose_to_imu_indices_;
    std::unordered_map<int, std::vector<PriorConstraint>> priors_;

    static bool isValidId(int id);
    static bool isFinite(const Eigen::Vector2f& v);
    static bool isFinite(const Eigen::Vector3f& v);
    static bool isFinite(const Eigen::Matrix3f& m);

    static void validateMeasurementSegment(const ImuEdgeInput& edge_input);

    void validateObservation(const ObservationInput& obs_input) const;
    CameraObservation makeCameraObservation(const ObservationInput& obs_input) const;
    PriorConstraint makePriorConstraint(const PriorInput& prior_input) const;

    std::vector<ImuConstraint> collectImuConstraints(int pose_id, std::vector<std::size_t>* consumed_indices);
    std::vector<PriorConstraint> collectPriorConstraints(int pose_id) const;
    bool hasUnusedImu(int pose_id) const;
    void markImuConstraintUsed(std::size_t idx);
};

#endif // ORBSLAM3_EXPORTER_H

```

```cpp
#include "ORBSLAM3_Exporter.h"
#include <stdexcept>
#include <cmath>
#include <unordered_set>

namespace {
constexpr float kEpsilon = 1e-9f;

template <typename T>
bool isFiniteScalar(T value) {
    return std::isfinite(static_cast<double>(value));
}
} // namespace

ORBSLAM3Exporter::ORBSLAM3Exporter(VariableEliminationDriver& driver)
    : driver_(driver) {}

void ORBSLAM3Exporter::addObservation(const ObservationInput& obs_input) {
    validateObservation(obs_input);
    CameraObservation obs = makeCameraObservation(obs_input);
    landmark_observations_[obs.landmark_id].push_back(obs);
}

void ORBSLAM3Exporter::addObservations(const std::vector<ObservationInput>& obs_inputs) {
    for (const auto& obs : obs_inputs) {
        addObservation(obs);
    }
}

void ORBSLAM3Exporter::addImuEdge(const ImuEdgeInput& edge_input) {
    if (!isValidId(edge_input.pose_i_id) || !isValidId(edge_input.pose_j_id)) {
        throw std::invalid_argument("IMU edge pose ids must be non-negative");
    }
    if (edge_input.pose_i_id == edge_input.pose_j_id) {
        throw std::invalid_argument("IMU edge must connect two distinct poses");
    }
    validateMeasurementSegment(edge_input);

    ImuConstraint constraint = driver_.makeImuConstraint(
        edge_input.pose_i_id,
        edge_input.pose_j_id,
        edge_input.measurements,
        edge_input.dt_total
    );

    ImuConstraintEntry entry;
    entry.constraint = constraint;
    entry.used = false;

    std::size_t idx = imu_constraints_.size();
    imu_constraints_.push_back(entry);
    pose_to_imu_indices_[constraint.pose_i_id].push_back(idx);
    pose_to_imu_indices_[constraint.pose_j_id].push_back(idx);
}

void ORBSLAM3Exporter::addImuEdges(const std::vector<ImuEdgeInput>& edge_inputs) {
    for (const auto& edge : edge_inputs) {
        addImuEdge(edge);
    }
}

void ORBSLAM3Exporter::addPrior(const PriorInput& prior_input) {
    if (!isValidId(prior_input.pose_id)) {
        throw std::invalid_argument("Prior pose id must be non-negative");
    }
    PriorConstraint prior = makePriorConstraint(prior_input);
    priors_[prior.pose_id].push_back(prior);
}

bool ORBSLAM3Exporter::eliminateLandmark(int landmark_id, float huber_delta) {
    auto it = landmark_observations_.find(landmark_id);
    if (it == landmark_observations_.end()) {
        return false;
    }
    if (it->second.empty()) {
        return false;
    }
    driver_.eliminateLandmark(landmark_id, it->second, huber_delta);
    landmark_observations_.erase(it);
    return true;
}

std::size_t ORBSLAM3Exporter::eliminateLandmarks(const std::vector<int>& landmark_ids, float huber_delta) {
    std::size_t count = 0;
    for (int id : landmark_ids) {
        if (eliminateLandmark(id, huber_delta)) {
            ++count;
        }
    }
    return count;
}

std::size_t ORBSLAM3Exporter::eliminateAllLandmarks(float huber_delta) {
    std::vector<int> ids = pendingLandmarkIds();
    return eliminateLandmarks(ids, huber_delta);
}

bool ORBSLAM3Exporter::eliminatePose(int pose_id) {
    std::vector<std::size_t> consumed_indices;
    std::vector<ImuConstraint> imu_constraints = collectImuConstraints(pose_id, &consumed_indices);
    std::vector<PriorConstraint> priors = collectPriorConstraints(pose_id);

    if (imu_constraints.empty() && priors.empty()) {
        return false;
    }

    driver_.eliminatePose(pose_id, imu_constraints, priors);

    for (std::size_t idx : consumed_indices) {
        markImuConstraintUsed(idx);
    }
    priors_.erase(pose_id);
    return true;
}

std::size_t ORBSLAM3Exporter::eliminatePoses(const std::vector<int>& pose_ids) {
    std::size_t count = 0;
    for (int id : pose_ids) {
        if (eliminatePose(id)) {
            ++count;
        }
    }
    return count;
}

std::vector<int> ORBSLAM3Exporter::pendingLandmarkIds() const {
    std::vector<int> ids;
    ids.reserve(landmark_observations_.size());
    for (const auto& kv : landmark_observations_) {
        ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::vector<int> ORBSLAM3Exporter::pendingPoseIds() const {
    std::unordered_set<int> unique_ids;
    for (const auto& kv : pose_to_imu_indices_) {
        if (hasUnusedImu(kv.first)) {
            unique_ids.insert(kv.first);
        }
    }
    for (const auto& kv : priors_) {
        unique_ids.insert(kv.first);
    }
    std::vector<int> ids(unique_ids.begin(), unique_ids.end());
    std::sort(ids.begin(), ids.end());
    return ids;
}

void ORBSLAM3Exporter::clear() {
    landmark_observations_.clear();
    imu_constraints_.clear();
    pose_to_imu_indices_.clear();
    priors_.clear();
}

bool ORBSLAM3Exporter::isValidId(int id) {
    return id >= 0;
}

bool ORBSLAM3Exporter::isFinite(const Eigen::Vector2f& v) {
    return isFiniteScalar(v.x()) && isFiniteScalar(v.y());
}

bool ORBSLAM3Exporter::isFinite(const Eigen::Vector3f& v) {
    return isFiniteScalar(v.x()) && isFiniteScalar(v.y()) && isFiniteScalar(v.z());
}

bool ORBSLAM3Exporter::isFinite(const Eigen::Matrix3f& m) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!isFiniteScalar(m(r, c))) {
                return false;
            }
        }
    }
    return true;
}

void ORBSLAM3Exporter::validateMeasurementSegment(const ImuEdgeInput& edge_input) {
    if (edge_input.measurements.empty()) {
        throw std::invalid_argument("IMU edge must contain measurements");
    }
    if (edge_input.dt_total <= 0.0) {
        throw std::invalid_argument("IMU edge dt_total must be positive");
    }
    for (std::size_t i = 0; i < edge_input.measurements.size(); ++i) {
        const auto& z = edge_input.measurements[i];
        if (z.dt <= 0.0) {
            throw std::invalid_argument("IMU measurement dt must be positive");
        }
        if (!std::isfinite(z.acc.x()) || !std::isfinite(z.acc.y()) || !std::isfinite(z.acc.z())) {
            throw std::invalid_argument("IMU measurement acc must be finite");
        }
        if (!std::isfinite(z.gyro.x()) || !std::isfinite(z.gyro.y()) || !std::isfinite(z.gyro.z())) {
            throw std::invalid_argument("IMU measurement gyro must be finite");
        }
    }
}

void ORBSLAM3Exporter::validateObservation(const ObservationInput& obs_input) const {
    if (!isValidId(obs_input.pose_id) || !isValidId(obs_input.landmark_id)) {
        throw std::invalid_argument("Observation ids must be non-negative");
    }
    if (!isFinite(obs_input.pixel)) {
        throw std::invalid_argument("Observation pixel must be finite");
    }
    if (!isFinite(obs_input.Rcw) || !isFinite(obs_input.tcw) || !isFinite(obs_input.landmark_w)) {
        throw std::invalid_argument("Observation pose/landmark entries must be finite");
    }
    if (!isFiniteScalar(obs_input.fx) || !isFiniteScalar(obs_input.fy) ||
        !isFiniteScalar(obs_input.cx) || !isFiniteScalar(obs_input.cy)) {
        throw std::invalid_argument("Observation intrinsics must be finite");
    }
    if (obs_input.fx < kEpsilon || obs_input.fy < kEpsilon) {
        throw std::invalid_argument("Observation intrinsics must be positive");
    }
    if (obs_input.sigma_pixel <= 0.0f || !isFiniteScalar(obs_input.sigma_pixel)) {
        throw std::invalid_argument("Observation sigma_pixel must be positive");
    }
}

CameraObservation ORBSLAM3Exporter::makeCameraObservation(const ObservationInput& obs_input) const {
    CameraObservation obs;
    obs.pose_id = obs_input.pose_id;
    obs.landmark_id = obs_input.landmark_id;
    obs.pixel = obs_input.pixel;
    obs.Rcw = obs_input.Rcw;
    obs.tcw = obs_input.tcw;
    obs.lw = obs_input.landmark_w;
    obs.fx = obs_input.fx;
    obs.fy = obs_input.fy;
    obs.cx = obs_input.cx;
    obs.cy = obs_input.cy;
    obs.sigma_pixel = obs_input.sigma_pixel;
    return obs;
}

PriorConstraint ORBSLAM3Exporter::makePriorConstraint(const PriorInput& prior_input) const {
    PriorConstraint prior;
    prior.pose_id = prior_input.pose_id;
    prior.Rp = prior_input.Rp;
    prior.tp = prior_input.tp;
    prior.sigma = prior_input.sigma;
    for (int i = 0; i < 6; ++i) {
        if (prior.sigma(i) <= 0.0f || !isFiniteScalar(prior.sigma(i))) {
            throw std::invalid_argument("Prior sigma must be positive");
        }
    }
    if (!isFinite(prior.tp) || !isFinite(prior.Rp)) {
        throw std::invalid_argument("Prior pose entries must be finite");
    }
    return prior;
}

std::vector<ImuConstraint> ORBSLAM3Exporter::collectImuConstraints(
    int pose_id,
    std::vector<std::size_t>* consumed_indices
) {
    std::vector<ImuConstraint> result;
    consumed_indices->clear();
    auto it = pose_to_imu_indices_.find(pose_id);
    if (it == pose_to_imu_indices_.end()) {
        return result;
    }
    for (std::size_t idx : it->second) {
        if (idx >= imu_constraints_.size()) {
            continue;
        }
        const auto& entry = imu_constraints_[idx];
        if (entry.used) {
            continue;
        }
        result.push_back(entry.constraint);
        consumed_indices->push_back(idx);
    }
    return result;
}

std::vector<PriorConstraint> ORBSLAM3Exporter::collectPriorConstraints(int pose_id) const {
    auto it = priors_.find(pose_id);
    if (it == priors_.end()) {
        return {};
    }
    return it->second;
}

bool ORBSLAM3Exporter::hasUnusedImu(int pose_id) const {
    auto it = pose_to_imu_indices_.find(pose_id);
    if (it == pose_to_imu_indices_.end()) {
        return false;
    }
    for (std::size_t idx : it->second) {
        if (idx < imu_constraints_.size() && !imu_constraints_[idx].used) {
            return true;
        }
    }
    return false;
}

void ORBSLAM3Exporter::markImuConstraintUsed(std::size_t idx) {
    if (idx < imu_constraints_.size()) {
        imu_constraints_[idx].used = true;
    }
}

```

## 结果
- 新增 `ORBSLAM3Exporter` 统一入口，定义了 observation / IMU / prior 三类输入结构与批量导入、消元、查询等接口，方便直接把 ORB-SLAM3 Tracking 拉取的数据推送到 `VariableEliminationDriver` 并追踪还未处理的 pose/landmark。（代码示例见下）
```15:65:QR/SW/ORBSLAM3_Exporter.h
class ORBSLAM3Exporter {
public:
    struct ObservationInput {
        int pose_id = -1;
        int landmark_id = -1;
        Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        Eigen::Vector3f landmark_w = Eigen::Vector3f::Zero();
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        float sigma_pixel = 1.0f;
    };
    struct ImuEdgeInput {
        int pose_i_id = -1;
        int pose_j_id = -1;
        std::vector<ImuPreprocessorGTSAM::ImuMeasurement> measurements;
        double dt_total = 0.0;
    };
    struct PriorInput {
        int pose_id = -1;
        Eigen::Matrix3f Rp = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tp = Eigen::Vector3f::Zero();
        Eigen::Matrix<float,6,1> sigma = Eigen::Matrix<float,6,1>::Constant(1.0f);
    };
    explicit ORBSLAM3Exporter(VariableEliminationDriver& driver);
    void addObservation(const ObservationInput& obs_input);
    void addImuEdge(const ImuEdgeInput& edge_input);
    void addPrior(const PriorInput& prior_input);
    bool eliminateLandmark(int landmark_id, float huber_delta = 2.5f);
    bool eliminatePose(int pose_id);
    std::vector<int> pendingLandmarkIds() const;
    std::vector<int> pendingPoseIds() const;
    void clear();
```
- 在实现侧加入严格的输入校验、IMU 预积分调用、消元后约束回收等逻辑，保证每个因子只被使用一次且可以自动统计 pending 集合，为滑窗调度提供决策依据。（核心逻辑见下）
```30:115:QR/SW/ORBSLAM3_Exporter.cpp
void ORBSLAM3Exporter::addImuEdge(const ImuEdgeInput& edge_input) {
    if (!isValidId(edge_input.pose_i_id) || !isValidId(edge_input.pose_j_id)) {
        throw std::invalid_argument("IMU edge pose ids must be non-negative");
    }
    if (edge_input.pose_i_id == edge_input.pose_j_id) {
        throw std::invalid_argument("IMU edge must connect two distinct poses");
    }
    validateMeasurementSegment(edge_input);

    ImuConstraint constraint = driver_.makeImuConstraint(
        edge_input.pose_i_id,
        edge_input.pose_j_id,
        edge_input.measurements,
        edge_input.dt_total
    );

    ImuConstraintEntry entry;
    entry.constraint = constraint;
    entry.used = false;

    std::size_t idx = imu_constraints_.size();
    imu_constraints_.push_back(entry);
    pose_to_imu_indices_[constraint.pose_i_id].push_back(idx);
    pose_to_imu_indices_[constraint.pose_j_id].push_back(idx);
}

bool ORBSLAM3Exporter::eliminatePose(int pose_id) {
    std::vector<std::size_t> consumed_indices;
    std::vector<ImuConstraint> imu_constraints = collectImuConstraints(pose_id, &consumed_indices);
    std::vector<PriorConstraint> priors = collectPriorConstraints(pose_id);

    if (imu_constraints.empty() && priors.empty()) {
        return false;
    }

    driver_.eliminatePose(pose_id, imu_constraints, priors);

    for (std::size_t idx : consumed_indices) {
        markImuConstraintUsed(idx);
    }
    priors_.erase(pose_id);
    return true;
}
```

## 后续建议
- 将该导出器嵌入 ORB-SLAM3 Tracking，按滑窗策略在关键帧生成后立即 `addObservation/addImuEdge/addPrior` 并依据 `pending*Ids()` 调用硬件消元。
- 结合 EuRoC 等数据集跑端到端验证，确认因子数量、硬件吞吐与预期一致（本次未运行自动化测试）。

```plaintext
class ORBSLAM3Exporter {
public:
    struct ObservationInput {
        int pose_id = -1;
        int landmark_id = -1;
        Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        Eigen::Vector3f landmark_w = Eigen::Vector3f::Zero();
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        float sigma_pixel = 1.0f;
    };
    struct ImuEdgeInput {
        int pose_i_id = -1;
        int pose_j_id = -1;
        std::vector<ImuPreprocessorGTSAM::ImuMeasurement> measurements;
        double dt_total = 0.0;
    };
    struct PriorInput {
        int pose_id = -1;
        Eigen::Matrix3f Rp = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tp = Eigen::Vector3f::Zero();
        Eigen::Matrix<float,6,1> sigma = Eigen::Matrix<float,6,1>::Constant(1.0f);
    };
    explicit ORBSLAM3Exporter(VariableEliminationDriver& driver);
    void addObservation(const ObservationInput& obs_input);
    void addImuEdge(const ImuEdgeInput& edge_input);
    void addPrior(const PriorInput& prior_input);
    bool eliminateLandmark(int landmark_id, float huber_delta = 2.5f);
    bool eliminatePose(int pose_id);
    std::vector<int> pendingLandmarkIds() const;
    std::vector<int> pendingPoseIds() const;
    void clear();
```

```cpp
void ORBSLAM3Exporter::addImuEdge(const ImuEdgeInput& edge_input) {
    if (!isValidId(edge_input.pose_i_id) || !isValidId(edge_input.pose_j_id)) {
        throw std::invalid_argument("IMU edge pose ids must be non-negative");
    }
    if (edge_input.pose_i_id == edge_input.pose_j_id) {
        throw std::invalid_argument("IMU edge must connect two distinct poses");
    }
    validateMeasurementSegment(edge_input);

    ImuConstraint constraint = driver_.makeImuConstraint(
        edge_input.pose_i_id,
        edge_input.pose_j_id,
        edge_input.measurements,
        edge_input.dt_total
    );

    ImuConstraintEntry entry;
    entry.constraint = constraint;
    entry.used = false;

    std::size_t idx = imu_constraints_.size();
    imu_constraints_.push_back(entry);
    pose_to_imu_indices_[constraint.pose_i_id].push_back(idx);
    pose_to_imu_indices_[constraint.pose_j_id].push_back(idx);
}

bool ORBSLAM3Exporter::eliminatePose(int pose_id) {
    std::vector<std::size_t> consumed_indices;
    std::vector<ImuConstraint> imu_constraints = collectImuConstraints(pose_id, &consumed_indices);
    std::vector<PriorConstraint> priors = collectPriorConstraints(pose_id);

    if (imu_constraints.empty() && priors.empty()) {
        return false;
    }

    driver_.eliminatePose(pose_id, imu_constraints, priors);

    for (std::size_t idx : consumed_indices) {
        markImuConstraintUsed(idx);
    }
    priors_.erase(pose_id);
    return true;
}
```

---

**User**

说明一下距离我实现自动化设计还要什么？

---

**Cursor**

离“自动化设计”还差几个关键拼图，建议按下面顺序推进：

- **接入导出器**：`ORBSLAM3Exporter` 已就绪（`QR/SW/ORBSLAM3_Exporter.*`），但要在 ORB-SLAM3 Tracking 内部把关键帧观测、IMU 段和先验实时 `addObservation / addImuEdge / addPrior`，并依据 `pending*Ids()` 驱动 `eliminateLandmark / eliminatePose`，把数据真正推到硬件链路。
- **数据集自动加载**：需要编写 EuRoC（或其他）loader，统一产出图像、IMU、初始化姿态，并把滑窗策略封装成循环脚本，实现“一键跑完一个数据集”。
- **流程调度/监控**：补充因子统计、耗时日志、错误兜底（例如 IMU 段缺失、姿态未设置时的告警），方便无人值守运行。
- **硬件侧缺口**：README 中仍留有 IMU Log/CORDIC、RowBuffer 自动列映射等 TODO，这些必须补齐才能让自动化跑通整条链。
- **验证与CI**：准备端到端测试或脚本（哪怕简单的 replay）并纳入自动化执行，确保每次改动都能回归完整流程。

只要把以上几个子系统串联，就能从“模块就绪”推进到真正的自动化设计/验证流程。需要我先帮你完成哪个步骤，直接告诉我。

---

**User**

先实现@MH01 的测试吧，我希望基于这个数据集进行测试。现在需要做的是我要输入这个Dataset，结果ORBSLAM3、GTSAM之类的处理之后，输出中间结果（也就是数据集作为输入，输出是我的硬件模块应该需要的输入）。我先跑通这里的测试。你先复述一下你对这段话的理解，我确认之后再开始

---

**Cursor**

你希望把 EuRoC 的 `MH01` 数据集（`/ORB_SLAM3/dataset/MH01`）作为输入，跑一条离线测试链路：先用 ORB-SLAM3 和 GTSAM 处理出关键帧、IMU 等前端结果，然后把这些结果转换成我们硬件模块（Axb+RowBuffer+SystolicArray）所需的中间因子格式（CameraObservation / ImuConstraint / PriorConstraint）。目标是尽快跑通这条“数据集 → 前端 → 导出器 → 硬件输入”的流程，以验证导出器和硬件接口在真实数据上的联调。确认无误后我再开始动手。

---

**User**

开始

---

**Cursor**

```cpp
/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef SYSTEM_H
#define SYSTEM_H


#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ImuTypes.h"
#include "Settings.h"


namespace ORB_SLAM3
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class Viewer;
class FrameDrawer;
class MapDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2,
        IMU_MONOCULAR=3,
        IMU_STEREO=4,
        IMU_RGBD=5,
    };

    // File type
    enum FileType{
        TEXT_FILE=0,
        BINARY_FILE=1,
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string());

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");


    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear Atlas or the active map)
    void Reset();
    void ResetActiveMap();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();
    bool isShutDown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    // For debugging
    double GetTimeFromIMUInit();
    bool isLost();
    bool isFinished();

    void ChangeDataset();

    float GetImageScale();

    Atlas* GetAtlas();
    const Atlas* GetAtlas() const;
    const Settings* GetSettings() const;

#ifdef REGISTER_TIMES
    void InsertRectTime(double& time);
    void InsertResizeTime(double& time);
    void InsertTrackTime(double& time);
#endif

private:

    void SaveAtlas(int type);
    bool LoadAtlas(int type);

    string CalculateCheckSum(string filename, int type);

    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;
    bool mbResetActiveMap;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Shutdown flag
    bool mbShutDown;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;

    //
    string mStrLoadAtlasFromFile;
    string mStrSaveAtlasToFile;

    string mStrVocabularyFilePath;

    Settings* settings_;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H

```

```cpp
/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <openssl/md5.h>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace ORB_SLAM3
{

Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer, const int initFr, const string &strSequence):
    mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false), mbResetActiveMap(false),
    mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false), mbShutDown(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;
    else if(mSensor==IMU_MONOCULAR)
        cout << "Monocular-Inertial" << endl;
    else if(mSensor==IMU_STEREO)
        cout << "Stereo-Inertial" << endl;
    else if(mSensor==IMU_RGBD)
        cout << "RGB-D-Inertial" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    cv::FileNode node = fsSettings["File.version"];
    if(!node.empty() && node.isString() && node.string() == "1.0"){
        settings_ = new Settings(strSettingsFile,mSensor);

        mStrLoadAtlasFromFile = settings_->atlasLoadFile();
        mStrSaveAtlasToFile = settings_->atlasSaveFile();

        cout << (*settings_) << endl;
    }
    else{
        settings_ = nullptr;
        cv::FileNode node = fsSettings["System.LoadAtlasFromFile"];
        if(!node.empty() && node.isString())
        {
            mStrLoadAtlasFromFile = (string)node;
        }

        node = fsSettings["System.SaveAtlasToFile"];
        if(!node.empty() && node.isString())
        {
            mStrSaveAtlasToFile = (string)node;
        }
    }

    node = fsSettings["loopClosing"];
    bool activeLC = true;
    if(!node.empty())
    {
        activeLC = static_cast<int>(fsSettings["loopClosing"]) != 0;
    }

    mStrVocabularyFilePath = strVocFile;

    bool loadedAtlas = false;

    if(mStrLoadAtlasFromFile.empty())
    {
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        //Create the Atlas
        cout << "Initialization of Atlas from scratch " << endl;
        mpAtlas = new Atlas(0);
    }
    else
    {
        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        cout << "Load File" << endl;

        // Load the file with an earlier session
        //clock_t start = clock();
        cout << "Initialization of Atlas from file: " << mStrLoadAtlasFromFile << endl;
        bool isRead = LoadAtlas(FileType::BINARY_FILE);

        if(!isRead)
        {
            cout << "Error to load the file, please try with other session file or vocabulary file" << endl;
            exit(-1);
        }
        //mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);


        //cout << "KF in DB: " << mpKeyFrameDatabase->mnNumKFs << "; words: " << mpKeyFrameDatabase->mnNumWords << endl;

        loadedAtlas = true;

        mpAtlas->CreateNewMap();

        //clock_t timeElapsed = clock() - start;
        //unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        //cout << "Binary file read in " << msElapsed << " ms" << endl;

        //usleep(10*1000*1000);
    }


    if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR || mSensor==IMU_RGBD)
        mpAtlas->SetInertialSensor();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpAtlas);
    mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile, settings_);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    cout << "Seq. Name: " << strSequence << endl;
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpAtlas, mpKeyFrameDatabase, strSettingsFile, mSensor, settings_, strSequence);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor==MONOCULAR || mSensor==IMU_MONOCULAR,
                                     mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD, strSequence);
    mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run,mpLocalMapper);
    mpLocalMapper->mInitFr = initFr;
    if(settings_)
        mpLocalMapper->mThFarPoints = settings_->thFarPoints();
    else
        mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];
    if(mpLocalMapper->mThFarPoints!=0)
    {
        cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
        mpLocalMapper->mbFarPoints = true;
    }
    else
        mpLocalMapper->mbFarPoints = false;

    //Initialize the Loop Closing thread and launch
    // mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR
    mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR, activeLC); // mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    //usleep(10*1000*1000);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    //if(false) // TODO
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile,settings_);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
        mpLoopCloser->mpViewer = mpViewer;
        mpViewer->both = mpFrameDrawer->both;
    }

    // Fix verbosity
    Verbose::SetTh(Verbose::VERBOSITY_QUIET);

}

Sophus::SE3f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    if(mSensor!=STEREO && mSensor!=IMU_STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to Stereo nor Stereo-Inertial." << endl;
        exit(-1);
    }

    cv::Mat imLeftToFeed, imRightToFeed;
    if(settings_ && settings_->needToRectify()){
        cv::Mat M1l = settings_->M1l();
        cv::Mat M2l = settings_->M2l();
        cv::Mat M1r = settings_->M1r();
        cv::Mat M2r = settings_->M2r();

        cv::remap(imLeft, imLeftToFeed, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(imRight, imRightToFeed, M1r, M2r, cv::INTER_LINEAR);
    }
    else if(settings_ && settings_->needToResize()){
        cv::resize(imLeft,imLeftToFeed,settings_->newImSize());
        cv::resize(imRight,imRightToFeed,settings_->newImSize());
    }
    else{
        imLeftToFeed = imLeft.clone();
        imRightToFeed = imRight.clone();
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_STEREO)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    // std::cout << "start GrabImageStereo" << std::endl;
    Sophus::SE3f Tcw = mpTracker->GrabImageStereo(imLeftToFeed,imRightToFeed,timestamp,filename);

    // std::cout << "out grabber" << std::endl;

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

Sophus::SE3f System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{
    if(mSensor!=RGBD  && mSensor!=IMU_RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    cv::Mat imDepthToFeed = depthmap.clone();
    if(settings_ && settings_->needToResize()){
        cv::Mat resizedIm;
        cv::resize(im,resizedIm,settings_->newImSize());
        imToFeed = resizedIm;

        cv::resize(depthmap,imDepthToFeed,settings_->newImSize());
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_RGBD)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageRGBD(imToFeed,imDepthToFeed,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

Sophus::SE3f System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas, string filename)
{

    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbShutDown)
            return Sophus::SE3f();
    }

    if(mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial." << endl;
        exit(-1);
    }

    cv::Mat imToFeed = im.clone();
    if(settings_ && settings_->needToResize()){
        cv::Mat resizedIm;
        cv::resize(im,resizedIm,settings_->newImSize());
        imToFeed = resizedIm;
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbReset)
        {
            mpTracker->Reset();
            mbReset = false;
            mbResetActiveMap = false;
        }
        else if(mbResetActiveMap)
        {
            cout << "SYSTEM-> Reseting active map in monocular case" << endl;
            mpTracker->ResetActiveMap();
            mbResetActiveMap = false;
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
        for(size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
            mpTracker->GrabImuData(vImuMeas[i_imu]);

    Sophus::SE3f Tcw = mpTracker->GrabImageMonocular(imToFeed,timestamp,filename);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}



void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpAtlas->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::ResetActiveMap()
{
    unique_lock<mutex> lock(mMutexReset);
    mbResetActiveMap = true;
}

void System::Shutdown()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbShutDown = true;
    }

    cout << "Shutdown" << endl;

    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    /*if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }*/

    // Wait until all thread have effectively stopped
    /*while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        if(!mpLocalMapper->isFinished())
            cout << "mpLocalMapper is not finished" << endl;*/
        /*if(!mpLoopCloser->isFinished())
            cout << "mpLoopCloser is not finished" << endl;
        if(mpLoopCloser->isRunningGBA()){
            cout << "mpLoopCloser is running GBA" << endl;
            cout << "break anyway..." << endl;
            break;
        }*/
        /*usleep(5000);
    }*/

    if(!mStrSaveAtlasToFile.empty())
    {
        Verbose::PrintMess("Atlas saving to file " + mStrSaveAtlasToFile, Verbose::VERBOSITY_NORMAL);
        SaveAtlas(FileType::BINARY_FILE);
    }

    /*if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");*/

#ifdef REGISTER_TIMES
    mpTracker->PrintTimeStats();
#endif


}

bool System::isShutDown() {
    unique_lock<mutex> lock(mMutexReset);
    return mbShutDown;
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();

        Eigen::Vector3f twc = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
    }
    f.close();
    // cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Eigen::Quaternionf q = Twc.unit_quaternion();
        Eigen::Vector3f t = Twc.translation();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t(0) << " " << t(1) << " " << t(2)
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

    }

    f.close();
}

void System::SaveTrajectoryEuRoC(const string &filename)
{

    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    /*if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    int numMaxKFs = 0;
    Map* pBiggerMap;
    std::cout << "There are " << std::to_string(vpMaps.size()) << " maps in the atlas" << std::endl;
    for(Map* pMap :vpMaps)
    {
        std::cout << "  Map " << std::to_string(pMap->GetId()) << " has " << std::to_string(pMap->GetAllKeyFrames().size()) << " KFs" << std::endl;
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b depending on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

void System::SaveTrajectoryEuRoC(const string &filename, Map* pMap)
{

    cout << endl << "Saving trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;
    /*if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }*/

    int numMaxKFs = 0;

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose();
    else
        Twb = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(auto lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;

        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = (pKF->mImuCalib.mTbc * (*lit) * Trw).inverse();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Twc = ((*lit)*Trw).inverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f twc = Twc.translation();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}

/*void System::SaveTrajectoryEuRoC(const string &filename)
{

    cout << endl << "Saving trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
        return;
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
    if (mSensor==IMU_MONOCULAR || mSensor==IMU_STEREO || mSensor==IMU_RGBD)
        Twb = vpKFs[0]->GetImuPose_();
    else
        Twb = vpKFs[0]->GetPoseInverse_();

    ofstream f;
    f.open(filename.c_str());
    // cout << "file open" << endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();

    //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
    //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
    //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
    //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        //cout << "1" << endl;
        if(*lbL)
            continue;


        KeyFrame* pKF = *lRit;
        //cout << "KF: " << pKF->mnId << endl;

        Sophus::SE3f Trw;

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        if (!pKF)
            continue;

        //cout << "2.5" << endl;

        while(pKF->isBad())
        {
            //cout << " 2.bad" << endl;
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
            //cout << "--Parent KF: " << pKF->mnId << endl;
        }

        if(!pKF || pKF->GetMap() != pBiggerMap)
        {
            //cout << "--Parent KF is from another map" << endl;
            continue;
        }

        //cout << "3" << endl;

        Trw = Trw * pKF->GetPose()*Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

        // cout << "4" << endl;


        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Tbw = pKF->mImuCalib.Tbc_ * (*lit) * Trw;
            Sophus::SE3f Twb = Tbw.inverse();

            Eigen::Vector3f twb = Twb.translation();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        else
        {
            Sophus::SE3f Tcw = (*lit) * Trw;
            Sophus::SE3f Twc = Tcw.inverse();

            Eigen::Vector3f twc = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            f << setprecision(6) << 1e9*(*lT) << " " <<  setprecision(9) << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }

        // cout << "5" << endl;
    }
    //cout << "end saving trajectory" << endl;
    f.close();
    cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
}*/


/*void System::SaveKeyFrameTrajectoryEuRoC_old(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            cv::Mat R = pKF->GetImuRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat twb = pKF->GetImuPosition();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb.at<float>(0) << " " << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        }
        else
        {
            cv::Mat R = pKF->GetRotation();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
    }
    f.close();
}*/

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBiggerMap;
    int numMaxKFs = 0;
    for(Map* pMap :vpMaps)
    {
        if(pMap && pMap->GetAllKeyFrames().size() > numMaxKFs)
        {
            numMaxKFs = pMap->GetAllKeyFrames().size();
            pBiggerMap = pMap;
        }
    }

    if(!pBiggerMap)
    {
        std::cout << "There is not a map!!" << std::endl;
        return;
    }

    vector<KeyFrame*> vpKFs = pBiggerMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

void System::SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap)
{
    cout << endl << "Saving keyframe trajectory of map " << pMap->GetId() << " to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

        if(!pKF || pKF->isBad())
            continue;
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO || mSensor==IMU_RGBD)
        {
            Sophus::SE3f Twb = pKF->GetImuPose();
            Eigen::Quaternionf q = Twb.unit_quaternion();
            Eigen::Vector3f twb = Twb.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp  << " " <<  setprecision(9) << twb(0) << " " << twb(1) << " " << twb(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

        }
        else
        {
            Sophus::SE3f Twc = pKF->GetPoseInverse();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            Eigen::Vector3f t = Twc.translation();
            f << setprecision(6) << 1e9*pKF->mTimeStamp << " " <<  setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
    }
    f.close();
}

/*void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
            Trw = Trw * Converter::toCvMat(pKF->mTcp.matrix());
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPoseCv() * Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
}*/

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    Sophus::SE3f Tow = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<Sophus::SE3f>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM3::KeyFrame* pKF = *lRit;

        Sophus::SE3f Trw;

        if(!pKF)
            continue;

        while(pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Tow;

        Sophus::SE3f Tcw = (*lit) * Trw;
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.rotationMatrix();
        Eigen::Vector3f twc = Twc.translation();

        f << setprecision(9) << Rwc(0,0) << " " << Rwc(0,1)  << " " << Rwc(0,2) << " "  << twc(0) << " " <<
             Rwc(1,0) << " " << Rwc(1,1)  << " " << Rwc(1,2) << " "  << twc(1) << " " <<
             Rwc(2,0) << " " << Rwc(2,1)  << " " << Rwc(2,2) << " "  << twc(2) << endl;
    }
    f.close();
}


void System::SaveDebugData(const int &initIdx)
{
    // 0. Save initialization trajectory
    SaveTrajectoryEuRoC("init_FrameTrajectoy_" +to_string(mpLocalMapper->mInitSect)+ "_" + to_string(initIdx)+".txt");

    // 1. Save scale
    ofstream f;
    f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mScale << endl;
    f.close();

    // 2. Save gravity direction
    f.open("init_GDir_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mRwg(0,0) << "," << mpLocalMapper->mRwg(0,1) << "," << mpLocalMapper->mRwg(0,2) << endl;
    f << mpLocalMapper->mRwg(1,0) << "," << mpLocalMapper->mRwg(1,1) << "," << mpLocalMapper->mRwg(1,2) << endl;
    f << mpLocalMapper->mRwg(2,0) << "," << mpLocalMapper->mRwg(2,1) << "," << mpLocalMapper->mRwg(2,2) << endl;
    f.close();

    // 3. Save computational cost
    f.open("init_CompCost_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mCostTime << endl;
    f.close();

    // 4. Save biases
    f.open("init_Biases_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
    f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
    f.close();

    // 5. Save covariance matrix
    f.open("init_CovMatrix_" +to_string(mpLocalMapper->mInitSect)+ "_" +to_string(initIdx)+".txt", ios_base::app);
    f << fixed;
    for(int i=0; i<mpLocalMapper->mcovInertial.rows(); i++)
    {
        for(int j=0; j<mpLocalMapper->mcovInertial.cols(); j++)
        {
            if(j!=0)
                f << ",";
            f << setprecision(15) << mpLocalMapper->mcovInertial(i,j);
        }
        f << endl;
    }
    f.close();

    // 6. Save initialization time
    f.open("init_Time_" +to_string(mpLocalMapper->mInitSect)+ ".txt", ios_base::app);
    f << fixed;
    f << mpLocalMapper->mInitTime << endl;
    f.close();
}


int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

double System::GetTimeFromIMUInit()
{
    double aux = mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    if ((aux>0.) && mpAtlas->isImuInitialized())
        return mpLocalMapper->GetCurrKFTime()-mpLocalMapper->mFirstTs;
    else
        return 0.f;
}

bool System::isLost()
{
    if (!mpAtlas->isImuInitialized())
        return false;
    else
    {
        if ((mpTracker->mState==Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
            return true;
        else
            return false;
    }
}


bool System::isFinished()
{
    return (GetTimeFromIMUInit()>0.1);
}

void System::ChangeDataset()
{
    if(mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12)
    {
        mpTracker->ResetActiveMap();
    }
    else
    {
        mpTracker->CreateMapInAtlas();
    }

    mpTracker->NewDataset();
}

float System::GetImageScale()
{
    return mpTracker->GetImageScale();
}

Atlas* System::GetAtlas()
{
    return mpAtlas;
}

const Atlas* System::GetAtlas() const
{
    return mpAtlas;
}

const Settings* System::GetSettings() const
{
    return settings_;
}

#ifdef REGISTER_TIMES
void System::InsertRectTime(double& time)
{
    mpTracker->vdRectStereo_ms.push_back(time);
}

void System::InsertResizeTime(double& time)
{
    mpTracker->vdResizeImage_ms.push_back(time);
}

void System::InsertTrackTime(double& time)
{
    mpTracker->vdTrackTotal_ms.push_back(time);
}
#endif

void System::SaveAtlas(int type){
    if(!mStrSaveAtlasToFile.empty())
    {
        //clock_t start = clock();

        // Save the current session
        mpAtlas->PreSave();

        string pathSaveFileName = "./";
        pathSaveFileName = pathSaveFileName.append(mStrSaveAtlasToFile);
        pathSaveFileName = pathSaveFileName.append(".osa");

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found+1);

        if(type == TEXT_FILE) // File text
        {
            cout << "Starting to write the save text file " << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            cout << "End to write the save text file" << endl;
        }
        else if(type == BINARY_FILE) // File binary
        {
            cout << "Starting to write the save binary file" << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            cout << "End to write save binary file" << endl;
        }
    }
}

bool System::LoadAtlas(int type)
{
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    string pathLoadFileName = "./";
    pathLoadFileName = pathLoadFileName.append(mStrLoadAtlasFromFile);
    pathLoadFileName = pathLoadFileName.append(".osa");

    if(type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save text file " << endl;
        isRead = true;
    }
    else if(type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file"  << endl;
        std::ifstream ifs(pathLoadFileName, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if(isRead)
    {
        //Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);

        if(strInputVocabularyChecksum.compare(strVocChecksum) != 0)
        {
            cout << "The vocabulary load isn't the same which the load session was created " << endl;
            cout << "-Vocabulary name: " << strFileVoc << endl;
            return false; // Both are differents
        }

        mpAtlas->SetKeyFrameDababase(mpKeyFrameDatabase);
        mpAtlas->SetORBVocabulary(mpVocabulary);
        mpAtlas->PostLoad();

        return true;
    }
    return false;
}

string System::CalculateCheckSum(string filename, int type)
{
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if(type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if ( !f.is_open() )
    {
        cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init (&md5Context);
    while ( int count = f.readsome(buffer, sizeof(buffer)))
    {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context );

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        char aux[10];
        sprintf(aux,"%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}

} //namespace ORB_SLAM


```

```cpp
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Atlas.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "System.h"

using namespace std;

namespace {

struct ImageSequence {
    vector<string> image_paths;
    vector<double> timestamps;
};

struct RawImuSample {
    double timestamp = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

struct LoadedImuData {
    vector<double> timestamps;
    vector<cv::Point3f> acc;
    vector<cv::Point3f> gyro;
    vector<RawImuSample> raw_samples;
};

struct PoseInfo {
    ORB_SLAM3::KeyFrame* keyframe = nullptr;
    int pose_id = -1;
    double timestamp = 0.0;
};

struct CameraObservationRecord {
    int pose_id = -1;
    int landmark_id = -1;
    Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
    Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
    Eigen::Vector3f landmark_w = Eigen::Vector3f::Zero();
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float sigma = 1.0f;
};

struct ImuMeasurementRecord {
    double dt = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

struct ImuEdgeRecord {
    int pose_i = -1;
    int pose_j = -1;
    double start_time = 0.0;
    double end_time = 0.0;
    vector<ImuMeasurementRecord> measurements;
};

struct PriorRecord {
    int pose_id = -1;
    Eigen::Matrix3f Rp = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tp = Eigen::Vector3f::Zero();
    Eigen::Matrix<float,6,1> sigma = Eigen::Matrix<float,6,1>::Constant(1.0f);
};

bool EnsureDirectory(const string& path) {
    if (path.empty()) {
        return false;
    }
    struct stat info {};
    if (stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        return true;
    }

    string partial;
    size_t start = 0;
    if (!path.empty() && path[0] == '/') {
        partial = "/";
        start = 1;
    }

    while (start <= path.size()) {
        size_t sep = path.find('/', start);
        string segment = path.substr(start, sep - start);
        if (!segment.empty()) {
            if (!partial.empty() && partial.back() != '/') {
                partial += "/";
            }
            partial += segment;
            if (stat(partial.c_str(), &info) != 0) {
                if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
                    cerr << "Failed to create directory: " << partial << " (" << strerror(errno) << ")" << endl;
                    return false;
                }
            } else if (!S_ISDIR(info.st_mode)) {
                cerr << "Path exists but is not a directory: " << partial << endl;
                return false;
            }
        }
        if (sep == string::npos) {
            break;
        }
        start = sep + 1;
    }
    return true;
}

void LoadImages(const string& image_path, const string& timestamp_file, ImageSequence& sequence) {
    ifstream fTimes(timestamp_file);
    if (!fTimes.is_open()) {
        throw runtime_error("Cannot open timestamp file: " + timestamp_file);
    }

    sequence.image_paths.reserve(5000);
    sequence.timestamps.reserve(5000);

    string line;
    while (getline(fTimes, line)) {
        if (line.empty()) {
            continue;
        }
        stringstream ss(line);
        string token;
        ss >> token;
        if (token.empty()) {
            continue;
        }
        double t = 0.0;
        try {
            t = stod(token) / 1e9;
        } catch (const std::exception&) {
            continue;
        }
        sequence.image_paths.push_back(image_path + "/" + token + ".png");
        sequence.timestamps.push_back(t);
    }
}

LoadedImuData LoadImuData(const string& csv_path) {
    ifstream f(csv_path);
    if (!f.is_open()) {
        throw runtime_error("Cannot open IMU file: " + csv_path);
    }

    LoadedImuData data;
    data.timestamps.reserve(5000);
    data.acc.reserve(5000);
    data.gyro.reserve(5000);
    data.raw_samples.reserve(5000);

    string line;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<double> values;
        values.reserve(7);
        string token;
        stringstream ss(line);
        while (getline(ss, token, ',')) {
            if (!token.empty()) {
                try {
                    values.push_back(stod(token));
                } catch (const std::exception&) {
                    values.clear();
                    break;
                }
            }
        }
        if (values.size() != 7) {
            continue;
        }

        double timestamp = values[0] / 1e9;
        Eigen::Vector3d gyro(values[1], values[2], values[3]);
        Eigen::Vector3d acc(values[4], values[5], values[6]);

        data.timestamps.push_back(timestamp);
        data.acc.emplace_back(static_cast<float>(acc.x()), static_cast<float>(acc.y()), static_cast<float>(acc.z()));
        data.gyro.emplace_back(static_cast<float>(gyro.x()), static_cast<float>(gyro.y()), static_cast<float>(gyro.z()));

        RawImuSample sample;
        sample.timestamp = timestamp;
        sample.acc = acc;
        sample.gyro = gyro;
        data.raw_samples.push_back(sample);
    }

    return data;
}

Map* SelectBestMap(ORB_SLAM3::Atlas* atlas) {
    if (!atlas) {
        return nullptr;
    }
    vector<ORB_SLAM3::Map*> maps = atlas->GetAllMaps();
    if (maps.empty()) {
        return atlas->GetCurrentMap();
    }

    ORB_SLAM3::Map* best = nullptr;
    size_t max_kf = 0;
    for (auto* map : maps) {
        if (!map) {
            continue;
        }
        size_t count = map->GetAllKeyFrames().size();
        if (count > max_kf) {
            best = map;
            max_kf = count;
        }
    }
    return best ? best : atlas->GetCurrentMap();
}

vector<PoseInfo> ExtractPoses(ORB_SLAM3::Map* map) {
    vector<PoseInfo> poses;
    if (!map) {
        return poses;
    }
    vector<ORB_SLAM3::KeyFrame*> keyframes = map->GetAllKeyFrames();
    poses.reserve(keyframes.size());
    for (auto* kf : keyframes) {
        if (!kf || kf->isBad()) {
            continue;
        }
        PoseInfo info;
        info.keyframe = kf;
        info.timestamp = kf->mTimeStamp;
        poses.push_back(info);
    }
    sort(poses.begin(), poses.end(), [](const PoseInfo& a, const PoseInfo& b) {
        return a.timestamp < b.timestamp;
    });
    for (size_t i = 0; i < poses.size(); ++i) {
        poses[i].pose_id = static_cast<int>(i);
    }
    return poses;
}

vector<CameraObservationRecord> BuildObservations(const vector<PoseInfo>& poses) {
    vector<CameraObservationRecord> records;
    unordered_map<ORB_SLAM3::MapPoint*, int> landmark_ids;
    int next_landmark_id = 0;

    for (const auto& pose : poses) {
        if (!pose.keyframe) {
            continue;
        }
        const auto vp_map_points = pose.keyframe->GetMapPointMatches();
        const auto& keypoints = pose.keyframe->mvKeysUn;
        for (size_t idx = 0; idx < vp_map_points.size(); ++idx) {
            auto* mp = vp_map_points[idx];
            if (!mp || mp->isBad()) {
                continue;
            }
            if (idx >= keypoints.size()) {
                continue;
            }
            const auto [it, inserted] = landmark_ids.emplace(mp, next_landmark_id);
            if (inserted) {
                ++next_landmark_id;
            }

            CameraObservationRecord rec;
            rec.pose_id = pose.pose_id;
            rec.landmark_id = it->second;
            rec.pixel = Eigen::Vector2f(keypoints[idx].pt.x, keypoints[idx].pt.y);
            rec.Rcw = pose.keyframe->GetRotation();
            rec.tcw = pose.keyframe->GetTranslation();
            rec.landmark_w = mp->GetWorldPos();
            rec.fx = pose.keyframe->fx;
            rec.fy = pose.keyframe->fy;
            rec.cx = pose.keyframe->cx;
            rec.cy = pose.keyframe->cy;
            rec.sigma = 1.0f;
            records.push_back(rec);
        }
    }
    return records;
}

vector<ImuMeasurementRecord> SliceImuSegment(double t_start,
                                             double t_end,
                                             const vector<RawImuSample>& samples) {
    vector<ImuMeasurementRecord> segment;
    if (samples.empty() || t_end <= t_start) {
        return segment;
    }

    const auto comp = [](const RawImuSample& sample, double value) {
        return sample.timestamp < value;
    };
    size_t idx = 0;
    auto it = lower_bound(samples.begin(), samples.end(), t_start, comp);
    if (it == samples.begin()) {
        idx = 0;
    } else if (it == samples.end()) {
        idx = samples.size() - 1;
    } else {
        idx = static_cast<size_t>(distance(samples.begin(), it));
        if (samples[idx].timestamp > t_start && idx > 0) {
            --idx;
        }
    }

    double last_time = t_start;
    Eigen::Vector3d last_acc = samples[idx].acc;
    Eigen::Vector3d last_gyro = samples[idx].gyro;

    for (; idx < samples.size() && samples[idx].timestamp <= t_end; ++idx) {
        double current = samples[idx].timestamp;
        if (current < last_time) {
            continue;
        }
        double dt = current - last_time;
        if (dt > 1e-9) {
            segment.push_back({dt, last_acc, last_gyro});
            last_time = current;
        }
        last_acc = samples[idx].acc;
        last_gyro = samples[idx].gyro;
    }

    if (last_time < t_end) {
        double dt = t_end - last_time;
        if (dt > 1e-9) {
            segment.push_back({dt, last_acc, last_gyro});
        }
    }

    return segment;
}

vector<ImuEdgeRecord> BuildImuEdges(const vector<PoseInfo>& poses,
                                    const vector<RawImuSample>& samples) {
    vector<ImuEdgeRecord> edges;
    if (poses.size() < 2) {
        return edges;
    }

    edges.reserve(poses.size());
    for (size_t i = 1; i < poses.size(); ++i) {
        const auto& prev_pose = poses[i - 1];
        const auto& curr_pose = poses[i];
        vector<ImuMeasurementRecord> meas = SliceImuSegment(prev_pose.timestamp, curr_pose.timestamp, samples);
        if (meas.empty()) {
            continue;
        }
        ImuEdgeRecord edge;
        edge.pose_i = prev_pose.pose_id;
        edge.pose_j = curr_pose.pose_id;
        edge.start_time = prev_pose.timestamp;
        edge.end_time = curr_pose.timestamp;
        edge.measurements = std::move(meas);
        edges.push_back(std::move(edge));
    }
    return edges;
}

PriorRecord BuildPrior(const PoseInfo& pose) {
    PriorRecord prior;
    prior.pose_id = pose.pose_id;
    if (pose.keyframe) {
        prior.Rp = pose.keyframe->GetRotation();
        prior.tp = pose.keyframe->GetTranslation();
    }
    prior.sigma << 0.05f, 0.05f, 0.05f, 0.01f, 0.01f, 0.01f;
    return prior;
}

template <typename Derived>
void WriteVectorJson(ostream& os, const Eigen::MatrixBase<Derived>& v) {
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v(i);
        if (i + 1 < v.size()) {
            os << ", ";
        }
    }
    os << "]";
}

void WriteMatrix3Json(ostream& os, const Eigen::Matrix3f& m) {
    os << "[";
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            os << m(r, c);
            if (r != 2 || c != 2) {
                os << ", ";
            }
        }
    }
    os << "]";
}

void WriteCameraObservations(const string& filepath,
                             const vector<CameraObservationRecord>& records) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"observations\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        os << "    {\n";
        os << "      \"pose_id\": " << rec.pose_id << ",\n";
        os << "      \"landmark_id\": " << rec.landmark_id << ",\n";
        os << "      \"pixel\": ";
        WriteVectorJson(os, rec.pixel);
        os << ",\n";
        os << "      \"Rcw\": ";
        WriteMatrix3Json(os, rec.Rcw);
        os << ",\n";
        os << "      \"tcw\": ";
        WriteVectorJson(os, rec.tcw);
        os << ",\n";
        os << "      \"landmark_w\": ";
        WriteVectorJson(os, rec.landmark_w);
        os << ",\n";
        os << "      \"intrinsics\": [" << rec.fx << ", " << rec.fy << ", " << rec.cx << ", " << rec.cy << "],\n";
        os << "      \"sigma_pixel\": " << rec.sigma << "\n";
        os << "    }";
        if (i + 1 < records.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WriteImuEdges(const string& filepath, const vector<ImuEdgeRecord>& edges) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"imu_edges\": [\n";
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& edge = edges[i];
        os << "    {\n";
        os << "      \"pose_i\": " << edge.pose_i << ",\n";
        os << "      \"pose_j\": " << edge.pose_j << ",\n";
        os << "      \"start_time\": " << edge.start_time << ",\n";
        os << "      \"end_time\": " << edge.end_time << ",\n";
        os << "      \"measurements\": [\n";
        for (size_t j = 0; j < edge.measurements.size(); ++j) {
            const auto& meas = edge.measurements[j];
            os << "        {\n";
            os << "          \"dt\": " << meas.dt << ",\n";
            os << "          \"acc\": ";
            WriteVectorJson(os, meas.acc);
            os << ",\n";
            os << "          \"gyro\": ";
            WriteVectorJson(os, meas.gyro);
            os << "\n";
            os << "        }";
            if (j + 1 < edge.measurements.size()) {
                os << ",";
            }
            os << "\n";
        }
        os << "      ]\n";
        os << "    }";
        if (i + 1 < edges.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WritePriors(const string& filepath, const vector<PriorRecord>& priors) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"priors\": [\n";
    for (size_t i = 0; i < priors.size(); ++i) {
        const auto& prior = priors[i];
        os << "    {\n";
        os << "      \"pose_id\": " << prior.pose_id << ",\n";
        os << "      \"Rp\": ";
        WriteMatrix3Json(os, prior.Rp);
        os << ",\n";
        os << "      \"tp\": ";
        WriteVectorJson(os, prior.tp);
        os << ",\n";
        os << "      \"sigma\": ";
        WriteVectorJson(os, prior.sigma);
        os << "\n";
        os << "    }";
        if (i + 1 < priors.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WritePoses(const string& filepath, const vector<PoseInfo>& poses) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"poses\": [\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& pose = poses[i];
        Eigen::Matrix3f Rcw = pose.keyframe ? pose.keyframe->GetRotation() : Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = pose.keyframe ? pose.keyframe->GetTranslation() : Eigen::Vector3f::Zero();
        os << "    {\n";
        os << "      \"pose_id\": " << pose.pose_id << ",\n";
        os << "      \"timestamp\": " << pose.timestamp << ",\n";
        os << "      \"Rcw\": ";
        WriteMatrix3Json(os, Rcw);
        os << ",\n";
        os << "      \"tcw\": ";
        WriteVectorJson(os, tcw);
        os << "\n";
        os << "    }";
        if (i + 1 < poses.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WriteSummary(const string& filepath,
                  size_t pose_count,
                  size_t observation_count,
                  size_t imu_edge_count,
                  size_t prior_count,
                  const string& dataset_path) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << "{\n";
    os << "  \"dataset\": \"" << dataset_path << "\",\n";
    os << "  \"pose_count\": " << pose_count << ",\n";
    os << "  \"observation_count\": " << observation_count << ",\n";
    os << "  \"imu_edge_count\": " << imu_edge_count << ",\n";
    os << "  \"prior_count\": " << prior_count << "\n";
    os << "}\n";
}

void PrintUsage(const char* prog) {
    cerr << "Usage: " << prog << " path_to_vocabulary path_to_settings path_to_sequence "
         << "path_to_times_file output_directory\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        PrintUsage(argv[0]);
        return 1;
    }

    string voc_path = argv[1];
    string settings_path = argv[2];
    string sequence_path = argv[3];
    string times_file = argv[4];
    string output_dir = argv[5];

    try {
        ImageSequence sequence;
        LoadImages(sequence_path + "/mav0/cam0/data", times_file, sequence);
        if (sequence.image_paths.empty()) {
            throw runtime_error("No images found for sequence.");
        }

        LoadedImuData imu_data = LoadImuData(sequence_path + "/mav0/imu0/data.csv");
        if (imu_data.timestamps.empty()) {
            throw runtime_error("No IMU data found for sequence.");
        }

        vector<vector<ORB_SLAM3::IMU::Point>> imu_meas(1);
        vector<double> imu_ts = {imu_data.timestamps};
        vector<cv::Point3f> imu_acc = imu_data.acc;
        vector<cv::Point3f> imu_gyro = imu_data.gyro;

        vector<double>& imu_times = imu_data.timestamps;
        vector<cv::Point3f>& vAcc = imu_data.acc;
        vector<cv::Point3f>& vGyro = imu_data.gyro;

        ORB_SLAM3::System SLAM(voc_path, settings_path, ORB_SLAM3::System::IMU_MONOCULAR, false);
        float image_scale = SLAM.GetImageScale();

        int first_imu = 0;
        while (first_imu < static_cast<int>(imu_times.size()) && imu_times[first_imu] <= sequence.timestamps.front()) {
            first_imu++;
        }
        first_imu = max(0, first_imu - 1);

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vImuMeas.reserve(200);

        for (size_t ni = 0; ni < sequence.image_paths.size(); ++ni) {
            const string& image_file = sequence.image_paths[ni];
            cv::Mat im = cv::imread(image_file, cv::IMREAD_UNCHANGED);
            if (im.empty()) {
                cerr << "Failed to load image: " << image_file << endl;
                continue;
            }
            if (image_scale != 1.f) {
                int width = static_cast<int>(im.cols * image_scale);
                int height = static_cast<int>(im.rows * image_scale);
                cv::resize(im, im, cv::Size(width, height));
            }

            double tframe = sequence.timestamps[ni];
            vImuMeas.clear();
            if (ni > 0) {
                while (first_imu < static_cast<int>(imu_times.size()) && imu_times[first_imu] <= tframe) {
                    vImuMeas.emplace_back(
                        vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                        vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                        imu_times[first_imu]);
                    first_imu++;
                }
            }

            SLAM.TrackMonocular(im, tframe, vImuMeas);
        }

        SLAM.Shutdown();

        ORB_SLAM3::Atlas* atlas = SLAM.GetAtlas();
        ORB_SLAM3::Map* best_map = SelectBestMap(atlas);
        if (!best_map) {
            throw runtime_error("No map data available after running ORB-SLAM3.");
        }

        vector<PoseInfo> poses = ExtractPoses(best_map);
        if (poses.empty()) {
            throw runtime_error("No valid keyframes found in map.");
        }
        vector<CameraObservationRecord> observations = BuildObservations(poses);
        vector<ImuEdgeRecord> imu_edges = BuildImuEdges(poses, imu_data.raw_samples);
        vector<PriorRecord> priors = {BuildPrior(poses.front())};

        if (!EnsureDirectory(output_dir)) {
            throw runtime_error("Cannot create output directory: " + output_dir);
        }

        WritePoses(output_dir + "/poses.json", poses);
        WriteCameraObservations(output_dir + "/camera_observations.json", observations);
        WriteImuEdges(output_dir + "/imu_edges.json", imu_edges);
        WritePriors(output_dir + "/priors.json", priors);
        WriteSummary(output_dir + "/summary.json",
                     poses.size(),
                     observations.size(),
                     imu_edges.size(),
                     priors.size(),
                     sequence_path);

        cout << "Export completed. Output directory: " << output_dir << endl;
        cout << "Poses: " << poses.size()
             << ", Observations: " << observations.size()
             << ", IMU edges: " << imu_edges.size()
             << ", Priors: " << priors.size() << endl;
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}

```

```cpp
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Atlas.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "System.h"

using namespace std;

namespace {

struct ImageSequence {
    vector<string> image_paths;
    vector<double> timestamps;
};

struct RawImuSample {
    double timestamp = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

struct LoadedImuData {
    vector<double> timestamps;
    vector<cv::Point3f> acc;
    vector<cv::Point3f> gyro;
    vector<RawImuSample> raw_samples;
};

struct PoseInfo {
    ORB_SLAM3::KeyFrame* keyframe = nullptr;
    int pose_id = -1;
    double timestamp = 0.0;
};

struct CameraObservationRecord {
    int pose_id = -1;
    int landmark_id = -1;
    Eigen::Vector2f pixel = Eigen::Vector2f::Zero();
    Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
    Eigen::Vector3f landmark_w = Eigen::Vector3f::Zero();
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float sigma = 1.0f;
};

struct ImuMeasurementRecord {
    double dt = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

struct ImuEdgeRecord {
    int pose_i = -1;
    int pose_j = -1;
    double start_time = 0.0;
    double end_time = 0.0;
    vector<ImuMeasurementRecord> measurements;
};

struct PriorRecord {
    int pose_id = -1;
    Eigen::Matrix3f Rp = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tp = Eigen::Vector3f::Zero();
    Eigen::Matrix<float,6,1> sigma = Eigen::Matrix<float,6,1>::Constant(1.0f);
};

bool EnsureDirectory(const string& path) {
    if (path.empty()) {
        return false;
    }
    struct stat info {};
    if (stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode)) {
        return true;
    }

    string partial;
    size_t start = 0;
    if (!path.empty() && path[0] == '/') {
        partial = "/";
        start = 1;
    }

    while (start <= path.size()) {
        size_t sep = path.find('/', start);
        string segment = path.substr(start, sep - start);
        if (!segment.empty()) {
            if (!partial.empty() && partial.back() != '/') {
                partial += "/";
            }
            partial += segment;
            if (stat(partial.c_str(), &info) != 0) {
                if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
                    cerr << "Failed to create directory: " << partial << " (" << strerror(errno) << ")" << endl;
                    return false;
                }
            } else if (!S_ISDIR(info.st_mode)) {
                cerr << "Path exists but is not a directory: " << partial << endl;
                return false;
            }
        }
        if (sep == string::npos) {
            break;
        }
        start = sep + 1;
    }
    return true;
}

void LoadImages(const string& image_path, const string& timestamp_file, ImageSequence& sequence) {
    ifstream fTimes(timestamp_file);
    if (!fTimes.is_open()) {
        throw runtime_error("Cannot open timestamp file: " + timestamp_file);
    }

    sequence.image_paths.reserve(5000);
    sequence.timestamps.reserve(5000);

    string line;
    while (getline(fTimes, line)) {
        if (line.empty()) {
            continue;
        }
        stringstream ss(line);
        string token;
        ss >> token;
        if (token.empty()) {
            continue;
        }
        double t = 0.0;
        try {
            t = stod(token) / 1e9;
        } catch (const std::exception&) {
            continue;
        }
        sequence.image_paths.push_back(image_path + "/" + token + ".png");
        sequence.timestamps.push_back(t);
    }
}

LoadedImuData LoadImuData(const string& csv_path) {
    ifstream f(csv_path);
    if (!f.is_open()) {
        throw runtime_error("Cannot open IMU file: " + csv_path);
    }

    LoadedImuData data;
    data.timestamps.reserve(5000);
    data.acc.reserve(5000);
    data.gyro.reserve(5000);
    data.raw_samples.reserve(5000);

    string line;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        vector<double> values;
        values.reserve(7);
        string token;
        stringstream ss(line);
        while (getline(ss, token, ',')) {
            if (!token.empty()) {
                try {
                    values.push_back(stod(token));
                } catch (const std::exception&) {
                    values.clear();
                    break;
                }
            }
        }
        if (values.size() != 7) {
            continue;
        }

        double timestamp = values[0] / 1e9;
        Eigen::Vector3d gyro(values[1], values[2], values[3]);
        Eigen::Vector3d acc(values[4], values[5], values[6]);

        data.timestamps.push_back(timestamp);
        data.acc.emplace_back(static_cast<float>(acc.x()), static_cast<float>(acc.y()), static_cast<float>(acc.z()));
        data.gyro.emplace_back(static_cast<float>(gyro.x()), static_cast<float>(gyro.y()), static_cast<float>(gyro.z()));

        RawImuSample sample;
        sample.timestamp = timestamp;
        sample.acc = acc;
        sample.gyro = gyro;
        data.raw_samples.push_back(sample);
    }

    return data;
}

Map* SelectBestMap(ORB_SLAM3::Atlas* atlas) {
    if (!atlas) {
        return nullptr;
    }
    vector<ORB_SLAM3::Map*> maps = atlas->GetAllMaps();
    if (maps.empty()) {
        return atlas->GetCurrentMap();
    }

    ORB_SLAM3::Map* best = nullptr;
    size_t max_kf = 0;
    for (auto* map : maps) {
        if (!map) {
            continue;
        }
        size_t count = map->GetAllKeyFrames().size();
        if (count > max_kf) {
            best = map;
            max_kf = count;
        }
    }
    return best ? best : atlas->GetCurrentMap();
}

vector<PoseInfo> ExtractPoses(ORB_SLAM3::Map* map) {
    vector<PoseInfo> poses;
    if (!map) {
        return poses;
    }
    vector<ORB_SLAM3::KeyFrame*> keyframes = map->GetAllKeyFrames();
    poses.reserve(keyframes.size());
    for (auto* kf : keyframes) {
        if (!kf || kf->isBad()) {
            continue;
        }
        PoseInfo info;
        info.keyframe = kf;
        info.timestamp = kf->mTimeStamp;
        poses.push_back(info);
    }
    sort(poses.begin(), poses.end(), [](const PoseInfo& a, const PoseInfo& b) {
        return a.timestamp < b.timestamp;
    });
    for (size_t i = 0; i < poses.size(); ++i) {
        poses[i].pose_id = static_cast<int>(i);
    }
    return poses;
}

vector<CameraObservationRecord> BuildObservations(const vector<PoseInfo>& poses) {
    vector<CameraObservationRecord> records;
    unordered_map<ORB_SLAM3::MapPoint*, int> landmark_ids;
    int next_landmark_id = 0;

    for (const auto& pose : poses) {
        if (!pose.keyframe) {
            continue;
        }
        const auto vp_map_points = pose.keyframe->GetMapPointMatches();
        const auto& keypoints = pose.keyframe->mvKeysUn;
        for (size_t idx = 0; idx < vp_map_points.size(); ++idx) {
            auto* mp = vp_map_points[idx];
            if (!mp || mp->isBad()) {
                continue;
            }
            if (idx >= keypoints.size()) {
                continue;
            }
            const auto [it, inserted] = landmark_ids.emplace(mp, next_landmark_id);
            if (inserted) {
                ++next_landmark_id;
            }

            CameraObservationRecord rec;
            rec.pose_id = pose.pose_id;
            rec.landmark_id = it->second;
            rec.pixel = Eigen::Vector2f(keypoints[idx].pt.x, keypoints[idx].pt.y);
            rec.Rcw = pose.keyframe->GetRotation();
            rec.tcw = pose.keyframe->GetTranslation();
            rec.landmark_w = mp->GetWorldPos();
            rec.fx = pose.keyframe->fx;
            rec.fy = pose.keyframe->fy;
            rec.cx = pose.keyframe->cx;
            rec.cy = pose.keyframe->cy;
            rec.sigma = 1.0f;
            records.push_back(rec);
        }
    }
    return records;
}

vector<ImuMeasurementRecord> SliceImuSegment(double t_start,
                                             double t_end,
                                             const vector<RawImuSample>& samples) {
    vector<ImuMeasurementRecord> segment;
    if (samples.empty() || t_end <= t_start) {
        return segment;
    }

    const auto comp = [](const RawImuSample& sample, double value) {
        return sample.timestamp < value;
    };
    size_t idx = 0;
    auto it = lower_bound(samples.begin(), samples.end(), t_start, comp);
    if (it == samples.begin()) {
        idx = 0;
    } else if (it == samples.end()) {
        idx = samples.size() - 1;
    } else {
        idx = static_cast<size_t>(distance(samples.begin(), it));
        if (samples[idx].timestamp > t_start && idx > 0) {
            --idx;
        }
    }

    double last_time = t_start;
    Eigen::Vector3d last_acc = samples[idx].acc;
    Eigen::Vector3d last_gyro = samples[idx].gyro;

    for (; idx < samples.size() && samples[idx].timestamp <= t_end; ++idx) {
        double current = samples[idx].timestamp;
        if (current < last_time) {
            continue;
        }
        double dt = current - last_time;
        if (dt > 1e-9) {
            segment.push_back({dt, last_acc, last_gyro});
            last_time = current;
        }
        last_acc = samples[idx].acc;
        last_gyro = samples[idx].gyro;
    }

    if (last_time < t_end) {
        double dt = t_end - last_time;
        if (dt > 1e-9) {
            segment.push_back({dt, last_acc, last_gyro});
        }
    }

    return segment;
}

vector<ImuEdgeRecord> BuildImuEdges(const vector<PoseInfo>& poses,
                                    const vector<RawImuSample>& samples) {
    vector<ImuEdgeRecord> edges;
    if (poses.size() < 2) {
        return edges;
    }

    edges.reserve(poses.size());
    for (size_t i = 1; i < poses.size(); ++i) {
        const auto& prev_pose = poses[i - 1];
        const auto& curr_pose = poses[i];
        vector<ImuMeasurementRecord> meas = SliceImuSegment(prev_pose.timestamp, curr_pose.timestamp, samples);
        if (meas.empty()) {
            continue;
        }
        ImuEdgeRecord edge;
        edge.pose_i = prev_pose.pose_id;
        edge.pose_j = curr_pose.pose_id;
        edge.start_time = prev_pose.timestamp;
        edge.end_time = curr_pose.timestamp;
        edge.measurements = std::move(meas);
        edges.push_back(std::move(edge));
    }
    return edges;
}

PriorRecord BuildPrior(const PoseInfo& pose) {
    PriorRecord prior;
    prior.pose_id = pose.pose_id;
    if (pose.keyframe) {
        prior.Rp = pose.keyframe->GetRotation();
        prior.tp = pose.keyframe->GetTranslation();
    }
    prior.sigma << 0.05f, 0.05f, 0.05f, 0.01f, 0.01f, 0.01f;
    return prior;
}

template <typename Derived>
void WriteVectorJson(ostream& os, const Eigen::MatrixBase<Derived>& v) {
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v(i);
        if (i + 1 < v.size()) {
            os << ", ";
        }
    }
    os << "]";
}

void WriteMatrix3Json(ostream& os, const Eigen::Matrix3f& m) {
    os << "[";
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            os << m(r, c);
            if (r != 2 || c != 2) {
                os << ", ";
            }
        }
    }
    os << "]";
}

void WriteCameraObservations(const string& filepath,
                             const vector<CameraObservationRecord>& records) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"observations\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        os << "    {\n";
        os << "      \"pose_id\": " << rec.pose_id << ",\n";
        os << "      \"landmark_id\": " << rec.landmark_id << ",\n";
        os << "      \"pixel\": ";
        WriteVectorJson(os, rec.pixel);
        os << ",\n";
        os << "      \"Rcw\": ";
        WriteMatrix3Json(os, rec.Rcw);
        os << ",\n";
        os << "      \"tcw\": ";
        WriteVectorJson(os, rec.tcw);
        os << ",\n";
        os << "      \"landmark_w\": ";
        WriteVectorJson(os, rec.landmark_w);
        os << ",\n";
        os << "      \"intrinsics\": [" << rec.fx << ", " << rec.fy << ", " << rec.cx << ", " << rec.cy << "],\n";
        os << "      \"sigma_pixel\": " << rec.sigma << "\n";
        os << "    }";
        if (i + 1 < records.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WriteImuEdges(const string& filepath, const vector<ImuEdgeRecord>& edges) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"imu_edges\": [\n";
    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& edge = edges[i];
        os << "    {\n";
        os << "      \"pose_i\": " << edge.pose_i << ",\n";
        os << "      \"pose_j\": " << edge.pose_j << ",\n";
        os << "      \"start_time\": " << edge.start_time << ",\n";
        os << "      \"end_time\": " << edge.end_time << ",\n";
        os << "      \"measurements\": [\n";
        for (size_t j = 0; j < edge.measurements.size(); ++j) {
            const auto& meas = edge.measurements[j];
            os << "        {\n";
            os << "          \"dt\": " << meas.dt << ",\n";
            os << "          \"acc\": ";
            WriteVectorJson(os, meas.acc);
            os << ",\n";
            os << "          \"gyro\": ";
            WriteVectorJson(os, meas.gyro);
            os << "\n";
            os << "        }";
            if (j + 1 < edge.measurements.size()) {
                os << ",";
            }
            os << "\n";
        }
        os << "      ]\n";
        os << "    }";
        if (i + 1 < edges.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WritePriors(const string& filepath, const vector<PriorRecord>& priors) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"priors\": [\n";
    for (size_t i = 0; i < priors.size(); ++i) {
        const auto& prior = priors[i];
        os << "    {\n";
        os << "      \"pose_id\": " << prior.pose_id << ",\n";
        os << "      \"Rp\": ";
        WriteMatrix3Json(os, prior.Rp);
        os << ",\n";
        os << "      \"tp\": ";
        WriteVectorJson(os, prior.tp);
        os << ",\n";
        os << "      \"sigma\": ";
        WriteVectorJson(os, prior.sigma);
        os << "\n";
        os << "    }";
        if (i + 1 < priors.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WritePoses(const string& filepath, const vector<PoseInfo>& poses) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << fixed << setprecision(9);
    os << "{\n  \"poses\": [\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        const auto& pose = poses[i];
        Eigen::Matrix3f Rcw = pose.keyframe ? pose.keyframe->GetRotation() : Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = pose.keyframe ? pose.keyframe->GetTranslation() : Eigen::Vector3f::Zero();
        os << "    {\n";
        os << "      \"pose_id\": " << pose.pose_id << ",\n";
        os << "      \"timestamp\": " << pose.timestamp << ",\n";
        os << "      \"Rcw\": ";
        WriteMatrix3Json(os, Rcw);
        os << ",\n";
        os << "      \"tcw\": ";
        WriteVectorJson(os, tcw);
        os << "\n";
        os << "    }";
        if (i + 1 < poses.size()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n}\n";
}

void WriteSummary(const string& filepath,
                  size_t pose_count,
                  size_t observation_count,
                  size_t imu_edge_count,
                  size_t prior_count,
                  const string& dataset_path) {
    ofstream os(filepath);
    if (!os.is_open()) {
        throw runtime_error("Failed to open output file: " + filepath);
    }
    os << "{\n";
    os << "  \"dataset\": \"" << dataset_path << "\",\n";
    os << "  \"pose_count\": " << pose_count << ",\n";
    os << "  \"observation_count\": " << observation_count << ",\n";
    os << "  \"imu_edge_count\": " << imu_edge_count << ",\n";
    os << "  \"prior_count\": " << prior_count << "\n";
    os << "}\n";
}

void PrintUsage(const char* prog) {
    cerr << "Usage: " << prog << " path_to_vocabulary path_to_settings path_to_sequence "
         << "path_to_times_file output_directory\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        PrintUsage(argv[0]);
        return 1;
    }

    string voc_path = argv[1];
    string settings_path = argv[2];
    string sequence_path = argv[3];
    string times_file = argv[4];
    string output_dir = argv[5];

    try {
        ImageSequence sequence;
        LoadImages(sequence_path + "/mav0/cam0/data", times_file, sequence);
        if (sequence.image_paths.empty()) {
            throw runtime_error("No images found for sequence.");
        }

        LoadedImuData imu_data = LoadImuData(sequence_path + "/mav0/imu0/data.csv");
        if (imu_data.timestamps.empty()) {
            throw runtime_error("No IMU data found for sequence.");
        }

        const vector<double>& imu_times = imu_data.timestamps;
        const vector<cv::Point3f>& vAcc = imu_data.acc;
        const vector<cv::Point3f>& vGyro = imu_data.gyro;

        ORB_SLAM3::System SLAM(voc_path, settings_path, ORB_SLAM3::System::IMU_MONOCULAR, false);
        float image_scale = SLAM.GetImageScale();

        int first_imu = 0;
        while (first_imu < static_cast<int>(imu_times.size()) && imu_times[first_imu] <= sequence.timestamps.front()) {
            first_imu++;
        }
        first_imu = max(0, first_imu - 1);

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vImuMeas.reserve(200);

        for (size_t ni = 0; ni < sequence.image_paths.size(); ++ni) {
            const string& image_file = sequence.image_paths[ni];
            cv::Mat im = cv::imread(image_file, cv::IMREAD_UNCHANGED);
            if (im.empty()) {
                cerr << "Failed to load image: " << image_file << endl;
                continue;
            }
            if (image_scale != 1.f) {
                int width = static_cast<int>(im.cols * image_scale);
                int height = static_cast<int>(im.rows * image_scale);
                cv::resize(im, im, cv::Size(width, height));
            }

            double tframe = sequence.timestamps[ni];
            vImuMeas.clear();
            if (ni > 0) {
                while (first_imu < static_cast<int>(imu_times.size()) && imu_times[first_imu] <= tframe) {
                    vImuMeas.emplace_back(
                        vAcc[first_imu].x, vAcc[first_imu].y, vAcc[first_imu].z,
                        vGyro[first_imu].x, vGyro[first_imu].y, vGyro[first_imu].z,
                        imu_times[first_imu]);
                    first_imu++;
                }
            }

            SLAM.TrackMonocular(im, tframe, vImuMeas);
        }

        SLAM.Shutdown();

        ORB_SLAM3::Atlas* atlas = SLAM.GetAtlas();
        ORB_SLAM3::Map* best_map = SelectBestMap(atlas);
        if (!best_map) {
            throw runtime_error("No map data available after running ORB-SLAM3.");
        }

        vector<PoseInfo> poses = ExtractPoses(best_map);
        if (poses.empty()) {
            throw runtime_error("No valid keyframes found in map.");
        }
        vector<CameraObservationRecord> observations = BuildObservations(poses);
        vector<ImuEdgeRecord> imu_edges = BuildImuEdges(poses, imu_data.raw_samples);
        vector<PriorRecord> priors = {BuildPrior(poses.front())};

        if (!EnsureDirectory(output_dir)) {
            throw runtime_error("Cannot create output directory: " + output_dir);
        }

        WritePoses(output_dir + "/poses.json", poses);
        WriteCameraObservations(output_dir + "/camera_observations.json", observations);
        WriteImuEdges(output_dir + "/imu_edges.json", imu_edges);
        WritePriors(output_dir + "/priors.json", priors);
        WriteSummary(output_dir + "/summary.json",
                     poses.size(),
                     observations.size(),
                     imu_edges.size(),
                     priors.size(),
                     sequence_path);

        cout << "Export completed. Output directory: " << output_dir << endl;
        cout << "Poses: " << poses.size()
             << ", Observations: " << observations.size()
             << ", IMU edges: " << imu_edges.size()
             << ", Priors: " << priors.size() << endl;
    } catch (const exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}


```

```cmake
cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++14 support (required for sigslot and other dependencies)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC14)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.4 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# Build examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_hw_export
        Examples/Monocular-Inertial/mono_inertial_hw_export.cc)
target_link_libraries(mono_inertial_hw_export ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()

#Old examples

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D)

add_executable(rgbd_tum_old
        Examples_old/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i_old
            Examples_old/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i_old
            Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo)

add_executable(stereo_kitti_old
        Examples_old/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

add_executable(stereo_euroc_old
        Examples_old/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

add_executable(stereo_tum_vi_old
        Examples_old/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265_old
            Examples_old/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i_old
            Examples_old/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular)

add_executable(mono_tum_old
        Examples_old/Monocular/mono_tum.cc)
target_link_libraries(mono_tum_old ${PROJECT_NAME})

add_executable(mono_kitti_old
        Examples_old/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti_old ${PROJECT_NAME})

add_executable(mono_euroc_old
        Examples_old/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc_old ${PROJECT_NAME})

add_executable(mono_tum_vi_old
        Examples_old/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265_old
            Examples_old/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_realsense_D435i_old
            Examples_old/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Monocular-Inertial)

add_executable(mono_inertial_euroc_old
        Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi_old
        Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i_old
            Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples_old/Stereo-Inertial)

add_executable(stereo_inertial_euroc_old
        Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi_old
        Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i_old
            Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
endif()

```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
 ## 系统架构
 
 ```
 ORB-SLAM3前端   GTSAM IMU预处理
      ↓               ↓
    Camera因子    IMU因子     Prior因子
         ↓          ↓            ↓
         └──────────┴────────────┘
                    ↓
               我的ASIC硬件
 ```
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下执行常规 `cmake .. && make`。新的可执行文件 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会随标准目标一起生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可根据需要修改）  
+   - 数据集：解压到 `ORB_SLAM3/dataset/MH01`，目录需包含 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   在 `ORB_SLAM3/` 下执行：
+   ```
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   该程序会重放整条 MH01 序列，运行 ORB-SLAM3 Monocular-Inertial 前端，并在后处理阶段遍历 Atlas，按关键帧顺序收集 Camera/IMU/Prior 因子。
+
+4. **输出文件说明（均为 JSON）**  
+   - `poses.json`：每个 pose_id 对应的关键帧时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：`pose_id`-`landmark_id` 的像素观测，包含 `Rcw`、`tcw`、`landmark_w` 与相机内参  
+   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
+   - `priors.json`：当前实现给第一个 pose 一个先验（可按需要扩展）  
+   - `summary.json`：统计信息（因子数量、数据集路径等）
+
+5. **下一步**  
+   - 读取这些 JSON，构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，即可通过 `VariableEliminationDriver` 下发到 Axb + RowBuffer + SystolicArray。  
+   - 若只关注某个窗口，可在解析阶段筛选 `pose_id`。  
+   - 需要重复实验时可直接更换数据集或输出目录，无需再次修改代码。
*** End Patch
```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
 `````` 搬
 ORB-SLAM3前端   GTSAM IMU预处理
      ↓               ↓
    Camera因子    IMU因子     Prior因子
         ↓          ↓            ↓
         └──────────┴────────────┘
                    ↓
               我的ASIC硬件
 ``````
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下按常规流程 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会一并生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需要调整）  
+   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，保证存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   ```bash
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   程序会重放整条 MH01 序列，调用 Monocular-Inertial ORB-SLAM3 前端，并在结束后遍历 Atlas，按关键帧顺序提取 Camera/IMU/Prior 因子。
+
+4. **输出文件（JSON）**  
+   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
+   - `imu_edges.json`：相邻 pose 间的原始 IMU 序列（`dt`、`acc`、`gyro`），供后续 GTSAM 预积分使用  
+   - `priors.json`：对首帧姿态给定 6 DoF 先验（可扩展为多帧）  
+   - `summary.json`：统计信息（pose/因子数量、数据集路径等）
+
+5. **后续接入**  
+   - 解析上述 JSON，构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，即可交给 `VariableEliminationDriver` → Axb → RowBuffer → SystolicArray。  
+   - `imu_edges.json` 中提供的原始序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
+   - 若只需测试局部窗口，可在解析阶段筛选 `pose_id` 后再消元。
*** End Patch
```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
 ```
 ORB-SLAM3前端   GTSAM IMU预处理
      ↓               ↓
    Camera因子    IMU因子     Prior因子
         ↓          ↓            ↓
         └──────────┴────────────┘
                    ↓
               我的ASIC硬件
 ```
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下按常规流程 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
+   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   ```bash
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   程序会重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端估计关键帧与地图点，并在离线阶段遍历 Atlas，提取 Camera/IMU/Prior 因子。
+
+4. **输出文件（JSON）**  
+   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素点、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
+   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
+   - `priors.json`：对首个 pose 给定 6DoF 先验（可按需要扩展）  
+   - `summary.json`：统计信息（数量、数据集路径等）
+
+5. **接入硬件链路**  
+   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 消元即可驱动 `VariableEliminationDriver → Axb → RowBuffer → SystolicArray`。  
+   - `imu_edges.json` 中的序列直接交给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 做统一白化。  
+   - 若只需局部窗口，可在解析阶段过滤 `pose_id`，或只选取前 N 帧生成的因子。
*** End Patch
```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
 ORB-SLAM3前端   GTSAM IMU预处理
      ↓               ↓
    Camera因子    IMU因子     Prior因子
         ↓          ↓            ↓
         └──────────┴────────────┘
                    ↓
               我的ASIC硬件
 ```
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下按常规流程 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
+   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   ```bash
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   程序会重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端估计关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
+
+4. **输出文件（JSON）**  
+   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
+   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
+   - `priors.json`：对首个 pose 给定 6DoF 先验（可按需要扩展）  
+   - `summary.json`：统计信息（数量、数据集路径等）
+
+5. **接入硬件链路**  
+   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 消元即可驱动 `VariableEliminationDriver → Axb → RowBuffer → SystolicArray`。  
+   - `imu_edges.json` 中的序列直接交给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 做统一白化。  
+   - 若只需局部窗口，可在解析阶段筛选 `pose_id`，或仅保留滑窗内的因子。
*** End Patch
```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
-# 硬件加速因子图优化系统
-
-## 系统架构
-
-```
-ORB-SLAM3前端   GTSAM IMU预处理
-     ↓               ↓
-   Camera因子    IMU因子     Prior因子
-        ↓          ↓            ↓
-        └──────────┴────────────┘
-                   ↓
-              我的ASIC硬件
-```
+# 硬件加速因子图优化系统
+
+## 系统架构
+
+```
+ORB-SLAM3前端   GTSAM IMU预处理
+     ↓               ↓
+   Camera因子    IMU因子     Prior因子
+        ↓          ↓            ↓
+        └──────────┴────────────┘
+                   ↓
+              我的ASIC硬件
+```
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
+   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   ```bash
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧和地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
+
+4. **输出文件（JSON）**  
+   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
+   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
+   - `priors.json`：对首个 pose 提供 6DoF 先验（可拓展到更多帧）  
+   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
+
+5. **接入硬件链路**  
+   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
+   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
+   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或保留前 N 帧进行快速验证。
*** End Patch
```

```markdown
# 硬件加速因子图优化系统

## 系统架构

```
ORB-SLAM3前端   GTSAM IMU预处理
     ↓               ↓
   Camera因子    IMU因子     Prior因子
        ↓          ↓            ↓
        └──────────┴────────────┘
                   ↓
              我的ASIC硬件
```

## EuRoC MH01 因子自动导出流程

1. **编译 ORB-SLAM3**  
   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。

2. **准备输入**  
   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`

3. **运行导出器**  
   ```bash
   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
       Vocabulary/ORBvoc.txt \
       Examples/Monocular-Inertial/EuRoC.yaml \
       dataset/MH01 \
       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
       /绝对路径/QR/output/mh01
   ```
   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。

4. **输出文件（JSON）**  
   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
   - `summary.json`：整体统计（pose/因子数量、数据集路径等）

5. **接入硬件链路**  
   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
# 硬件加速因子图优化系统

## 系统架构

```
ORB-SLAM3前端   GTSAM IMU预处理
     ↓               ↓
   Camera因子    IMU因子     Prior因子
        ↓          ↓            ↓
        └──────────┴────────────┘
                   ↓
              我的ASIC硬件

```

```markdown

## EuRoC MH01 因子自动导出流程

1. **编译 ORB-SLAM3**  
   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。

2. **准备输入**  
   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`

3. **运行导出器**  
   ```bash
   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
       Vocabulary/ORBvoc.txt \
       Examples/Monocular-Inertial/EuRoC.yaml \
       dataset/MH01 \
       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
       /绝对路径/QR/output/mh01
   ```
   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。

4. **输出文件（JSON）**  
   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
   - `summary.json`：整体统计（pose/因子数量、数据集路径等）

5. **接入硬件链路**  
   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
# 硬件加速因子图优化系统

## 系统架构

```
ORB-SLAM3前端   GTSAM IMU预处理
     ↓               ↓
   Camera因子    IMU因子     Prior因子
        ↓          ↓            ↓
        └──────────┴────────────┘
                   ↓
              我的ASIC硬件

```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
-## EuRoC MH01 因子自动导出流程
-
-1. **编译 ORB-SLAM3**  
-   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
-
-2. **准备输入**  
-   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
-   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
-   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
-   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
-
-3. **运行导出器**  
-   ```bash
-   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
-       Vocabulary/ORBvoc.txt \
-       Examples/Monocular-Inertial/EuRoC.yaml \
-       dataset/MH01 \
-       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
-       /绝对路径/QR/output/mh01
-   ```
-   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
-
-4. **输出文件（JSON）**  
-   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
-   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
-   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
-   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
-   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
-
-5. **接入硬件链路**  
-   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
-   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
-   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
-# 硬件加速因子图优化系统
-
-## 系统架构
-
-```
-ORB-SLAM3前端   GTSAM IMU预处理
-     ↓               ↓
-   Camera因子    IMU因子     Prior因子
-        ↓          ↓            ↓
-        └──────────┴────────────┘
-                   ↓
-              我的ASIC硬件
-```
+# 硬件加速因子图优化系统
+
+## 系统架构
+
+```
+ORB-SLAM3前端   GTSAM IMU预处理
+     ↓               ↓
+   Camera因子    IMU因子     Prior因子
+        ↓          ↓            ↓
+        └──────────┴────────────┘
+                   ↓
+              我的ASIC硬件
+```
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
+   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   ```bash
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
+
+4. **输出文件（JSON）**  
+   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
+   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
+   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
+   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
+
+5. **接入硬件链路**  
+   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
+   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
+   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
*** End Patch
```

```markdown
# 硬件加速因子图优化系统

## 系统架构

```
ORB-SLAM3前端   GTSAM IMU预处理
     ↓               ↓
   Camera因子    IMU因子     Prior因子
        ↓          ↓            ↓
        └──────────┴────────────┘
                   ↓
              我的ASIC硬件
```

## EuRoC MH01 因子自动导出流程

1. **编译 ORB-SLAM3**  
   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。

2. **准备输入**  
   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`

3. **运行导出器**  
   ```bash
   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
       Vocabulary/ORBvoc.txt \
       Examples/Monocular-Inertial/EuRoC.yaml \
       dataset/MH01 \
       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
       /绝对路径/QR/output/mh01
   ```
   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。

4. **输出文件（JSON）**  
   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
   - `summary.json`：整体统计（pose/因子数量、数据集路径等）

5. **接入硬件链路**  
   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。

## EuRoC MH01 因子自动导出流程

1. **编译 ORB-SLAM3**  
   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。

2. **准备输入**  
   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`

3. **运行导出器**  
   ```bash
   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
       Vocabulary/ORBvoc.txt \
       Examples/Monocular-Inertial/EuRoC.yaml \
       dataset/MH01 \
       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
       /绝对路径/QR/output/mh01
   ```
   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。

4. **输出文件（JSON）**  
   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
   - `summary.json`：整体统计（pose/因子数量、数据集路径等）

5. **接入硬件链路**  
   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
# 硬件加速因子图优化系统

## 系统架构

```
ORB-SLAM3前端   GTSAM IMU预处理
     ↓               ↓
   Camera因子    IMU因子     Prior因子
        ↓          ↓            ↓
        └──────────┴────────────┘
                   ↓
              我的ASIC硬件

```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
-# 硬件加速因子图优化系统
-
-## 系统架构
-
-```
-ORB-SLAM3前端   GTSAM IMU预处理
-     ↓               ↓
-   Camera因子    IMU因子     Prior因子
-        ↓          ↓            ↓
-        └──────────┴────────────┘
-                   ↓
-              我的ASIC硬件
-```
-
-## EuRoC MH01 因子自动导出流程
-
-1. **编译 ORB-SLAM3**  
-   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
-
-2. **准备输入**  
-   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
-   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
-   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
-   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
-
-3. **运行导出器**  
-   ```bash
-   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
-       Vocabulary/ORBvoc.txt \
-       Examples/Monocular-Inertial/EuRoC.yaml \
-       dataset/MH01 \
-       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
-       /绝对路径/QR/output/mh01
-   ```
-   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
-
-4. **输出文件（JSON）**  
-   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
-   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
-   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
-   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
-   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
-
-5. **接入硬件链路**  
-   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
-   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
-   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
-# 硬件加速因子图优化系统
-
-## 系统架构
-
-```
-ORB-SLAM3前端   GTSAM IMU预处理
-     ↓               ↓
-   Camera因子    IMU因子     Prior因子
-        ↓          ↓            ↓
-        └──────────┴────────────┘
-                   ↓
-              我的ASIC硬件
-```
*** End Patch
```

```markdown
*** Begin Patch
*** Update File: /Users/zhengliwen/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/study/HKUSTGZ/Systoilic/QR/README.md
@@
-# 硬件加速因子图优化系统
-
-## 系统架构
-
-```
-ORB-SLAM3前端   GTSAM IMU预处理
-     ↓               ↓
-   Camera因子    IMU因子     Prior因子
-        ↓          ↓            ↓
-        └──────────┴────────────┘
-                   ↓
-              我的ASIC硬件
-```
-
-## EuRoC MH01 因子自动导出流程
-
-1. **编译 ORB-SLAM3**  
-   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
-
-2. **准备输入**  
-   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
-   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
-   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
-   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
-
-3. **运行导出器**  
-   ```bash
-   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
-       Vocabulary/ORBvoc.txt \
-       Examples/Monocular-Inertial/EuRoC.yaml \
-       dataset/MH01 \
-       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
-       /绝对路径/QR/output/mh01
-   ```
-   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
-
-4. **输出文件（JSON）**  
-   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
-   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
-   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
-   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
-   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
-
-5. **接入硬件链路**  
-   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
-   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
-   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
-
-# 硬件加速因子图优化系统
-
-## 系统架构
-
-```
-ORB-SLAM3前端   GTSAM IMU预处理
-     ↓               ↓
-   Camera因子    IMU因子     Prior因子
-        ↓          ↓            ↓
-        └──────────┴────────────┘
-                   ↓
-              我的ASIC硬件
-```
-
-## EuRoC MH01 因子自动导出流程
-
-1. **编译 ORB-SLAM3**  
-   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
-
-2. **准备输入**  
-   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
-   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
-   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
-   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
-
-3. **运行导出器**  
-   ```bash
-   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
-       Vocabulary/ORBvoc.txt \
-       Examples/Monocular-Inertial/EuRoC.yaml \
-       dataset/MH01 \
-       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
-       /绝对路径/QR/output/mh01
-   ```
-   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
-
-4. **输出文件（JSON）**  
-   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
-   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
-   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
-   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
-   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
-
-5. **接入硬件链路**  
-   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
-   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
-   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
+# 硬件加速因子图优化系统
+
+## 系统架构
+
+```
+ORB-SLAM3前端   GTSAM IMU预处理
+     ↓               ↓
+   Camera因子    IMU因子     Prior因子
+        ↓          ↓            ↓
+        └──────────┴────────────┘
+                   ↓
+              我的ASIC硬件
+```
+
+## EuRoC MH01 因子自动导出流程
+
+1. **编译 ORB-SLAM3**  
+   在 `ORB_SLAM3/` 下执行 `mkdir build && cd build && cmake .. && make -j`，新的 `Examples/Monocular-Inertial/mono_inertial_hw_export` 会与标准示例一起生成。
+
+2. **准备输入**  
+   - 词典：`ORB_SLAM3/Vocabulary/ORBvoc.txt`  
+   - 配置：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC.yaml`（可按需调整）  
+   - 数据集：解压 `MH01` 至 `ORB_SLAM3/dataset/MH01`，确保存在 `mav0/cam0/data/*.png` 与 `mav0/imu0/data.csv`  
+   - 图像时间戳：`ORB_SLAM3/Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt`
+
+3. **运行导出器**  
+   ```bash
+   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
+       Vocabulary/ORBvoc.txt \
+       Examples/Monocular-Inertial/EuRoC.yaml \
+       dataset/MH01 \
+       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
+       /绝对路径/QR/output/mh01
+   ```
+   程序重放整条 MH01 序列，利用 Monocular-Inertial ORB-SLAM3 前端生成关键帧与地图点，并在离线阶段遍历 Atlas 提取 Camera / IMU / Prior 因子。
+
+4. **输出文件（JSON）**  
+   - `poses.json`：`pose_id`、时间戳、`Rcw`、`tcw`  
+   - `camera_observations.json`：每条观测的 `pose_id`、`landmark_id`、像素、`Rcw`、`tcw`、`landmark_w`、内参、`sigma_pixel`  
+   - `imu_edges.json`：相邻 pose 的原始 IMU 序列（`dt`、`acc`、`gyro`），用于后续 GTSAM 预积分  
+   - `priors.json`：首个 pose 的 6DoF 先验（可拓展为更多帧）  
+   - `summary.json`：整体统计（pose/因子数量、数据集路径等）
+
+5. **接入硬件链路**  
+   - 解析 JSON 后构造 `CameraObservation` / `ImuConstraint` / `PriorConstraint`，按 `pose_id` 调用 `VariableEliminationDriver` 即可驱动 `Axb → RowBuffer → SystolicArray`。  
+   - `imu_edges.json` 中的序列直接喂给 `VariableEliminationDriver::makeImuConstraint()`，由 `ImuPreprocessorGTSAM` 统一白化。  
+   - 若只需滑窗内因子，可在解析阶段筛选 `pose_id`，或仅保留前 N 帧进行快速验证。
*** End Patch
```

## 主要改动
- 在 `System.h/cc` 新增 `GetAtlas()`/`GetSettings()` 访问器，方便外部工具在 SLAM 停止后直接遍历 Atlas/Map 提取关键帧与地图点。
- 新增 `Examples/Monocular-Inertial/mono_inertial_hw_export.cc` 可执行程序：复用 EuRoC 数据加载流程，运行 ORB-SLAM3 Monocular-Inertial 前端，结束后按时间排序提取 pose、观测和原始 IMU 片段，序列化为 JSON（`poses.json`、`camera_observations.json`、`imu_edges.json`、`priors.json`、`summary.json`），为 `VariableEliminationDriver` 提供硬件输入所需的中间因子数据。
- 更新 `ORB_SLAM3/CMakeLists.txt` 将新工具纳入构建；重写 `QR/README.md`，给出 MH01 自动导出流程、命令示例与输出文件说明，方便复现实验。

## 使用说明
1. 在 `ORB_SLAM3` 根目录按常规 `mkdir build && cd build && cmake .. && make -j` 构建。
2. 准备 EuRoC 词典、配置、`dataset/MH01` 及时间戳文件。
3. 运行：
   ```
   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
       Vocabulary/ORBvoc.txt \
       Examples/Monocular-Inertial/EuRoC.yaml \
       dataset/MH01 \
       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
       /绝对路径/QR/output/mh01
   ```
   输出目录会生成上述 JSON，可直接解析成 `CameraObservation` / `ImuConstraint` / `PriorConstraint` 并送入 `VariableEliminationDriver`。

## Lint / 测试
- `read_lints` 指出 `opencv2/opencv_modules.hpp`、`openssl/md5.h` 等无法找到，原因是当前环境未装 OpenCV/OpenSSL 头文件；与新增代码逻辑无关，待本地配置好依赖即可消除。
- 未能在本地运行 ORB-SLAM3 全量测试（构建耗时+未装依赖）；建议在用户环境完成 `cmake && make` 后按 README 命令实测。

若后续需要把 JSON 解析直接接入 `ORBSLAM3_Exporter` 或限定滑窗，只需在解析阶段筛选 `pose_id` 并调用 `VariableEliminationDriver::makeImuConstraint()` 处理 `imu_edges.json` 中的原始段即可。

```plaintext
   ./Examples/Monocular-Inertial/mono_inertial_hw_export \
       Vocabulary/ORBvoc.txt \
       Examples/Monocular-Inertial/EuRoC.yaml \
       dataset/MH01 \
       Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt \
       /绝对路径/QR/output/mh01
```

---

**User**

我不理解为什么创建的是mono_inertial_hw_，我不是只用ORBSLAM3的camera吗？IMU不是利用GTSAM的预积分吗

---

**Cursor**

`mono_inertial_hw_export` 这个名字不是说你“必须”启用硬件 IMU，而是强调导出流程遵循 ORB-SLAM3 的 Monocular-Inertial 管线：  
- ORB-SLAM3 自身在 Monocular-Inertial 模式下就会把 camera tracking + 自带的 IMU 预积分结合起来（Tracking.cpp 里调用 `IMU::Preintegrated`）。我们只是复用它现成的逻辑，跑完整个 MH01。  
- 导出程序结束后只是把 ORB-SLAM3 已经算好的关键帧 / 地图点 / 原始 IMU 序列取出来。真正喂硬件之前，还是通过我们自己的 `VariableEliminationDriver::makeImuConstraint()` 去调用 GTSAM 的 `ImuPreprocessorGTSAM` 重新做一次预积分（这样能统一硬件侧需要的白化系数），camera 因子也一样走 `CameraObservation`。  

换句话说：  
1. ORB-SLAM3 → 负责前端估计 + 把相机/IMU 原始数据“打包”；  
2. GTSAM 预积分（`ImuPreprocessorGTSAM`） → 由我们导出器/驱动器统一计算，确保和硬件接口匹配；  
3. 你的硬件仍旧只收 Camera/IMU/Prior 因子，IMU 这段的预积分正是通过 GTSAM 模块完成的，并没有直接使用 ORB-SLAM3 的 IMU 输出。  

如果之后只想用纯相机数据，也可以把 EuRoC IMU CSV 这段不加载、`imu_edges.json` 置空就行。


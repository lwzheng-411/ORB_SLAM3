#include "HardwareAdapter.h"

#include "ORBSLAM3_Exporter.h"
#include "VariableEliminationDriver.h"
#include "io/JsonDumper.h"

#include <opencv2/core/core.hpp>
#include <set>
#include <stdexcept>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// 全局缓存：保存待写入的 JSON payload，供 Optimizer 在 LBA 完成后使用
DumpPayload g_pending_json_payload;
std::string g_pending_json_dir;
std::atomic<bool> g_has_pending_json{false};

namespace {
// 异步 JSON 写入队列
class AsyncJsonWriter {
public:
    static AsyncJsonWriter& Instance() {
        static AsyncJsonWriter instance;
        return instance;
    }

    void Enqueue(DumpPayload payload, std::string dir) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push({std::move(payload), std::move(dir)});
        }
        cv_.notify_one();
    }

    void Stop() {
        running_ = false;
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

private:
    AsyncJsonWriter() : running_(true) {
        worker_ = std::thread([this]() {
            while (running_ || !queue_.empty()) {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return !queue_.empty() || !running_; });
                while (!queue_.empty()) {
                    auto task = std::move(queue_.front());
                    queue_.pop();
                    lock.unlock();
                    DumpAllJson(task.first, task.second);
                    lock.lock();
                }
            }
        });
    }
    ~AsyncJsonWriter() { Stop(); }

    std::thread worker_;
    std::queue<std::pair<DumpPayload, std::string>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
};
} // namespace

// 异步写入 pending JSON（在 LBA 完成后调用）
void FlushPendingJson() {
    if (g_has_pending_json.exchange(false)) {
        AsyncJsonWriter::Instance().Enqueue(std::move(g_pending_json_payload), g_pending_json_dir);
        g_pending_json_payload = DumpPayload();  // 清空
        g_pending_json_dir.clear();
    }
}

namespace ORB_SLAM3 {
namespace {

Eigen::Matrix3f CvMatToEigen3f(const cv::Mat& R) {
    if (R.empty()) return Eigen::Matrix3f::Identity();
    CV_Assert(R.rows == 3 && R.cols == 3);
    Eigen::Matrix3f m;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m(r, c) = static_cast<float>(R.at<float>(r, c));
        }
    }
    return m;
}

Eigen::Vector3f CvMatToEigenVec3f(const cv::Mat& t) {
    if (t.empty()) return Eigen::Vector3f::Zero();
    CV_Assert(t.total() == 3);
    Eigen::Vector3f v;
    v << static_cast<float>(t.at<float>(0)),
         static_cast<float>(t.at<float>(1)),
         static_cast<float>(t.at<float>(2));
    return v;
}

bool IsKeyFrameInSet(KeyFrame* kf, const std::set<KeyFrame*>& s) {
    return s.find(kf) != s.end();
}

}  // namespace

HardwareAdapter::LocalBAResult HardwareAdapter::RunLocalBA(
    const LocalBAInput& input,
    const SolverSwitch& sw,
    const std::string& json_dir) {
    LocalBAResult result;
    if (!input.current_kf || !input.map) {
        return result;
    }

    VariableEliminationDriver driver;
    ORBSLAM3Exporter exporter(driver);

    // 填充因子
    FillExporter(input, driver, exporter);
    AddPriors(input, exporter);

    // JSON 导出移到 Optimizer 中与 baseline 一起输出，确保数量一致
    // 这里只保存 payload 到全局缓存，供 Optimizer 使用
    if (sw.dump_json) {
        std::string dir = !json_dir.empty() ? json_dir : sw.json_out_dir;
        if (dir.empty()) dir = "/tmp/orbslam3_hw_dump";
        // 保存 payload 供后续使用（在 LBA 完成后由 Optimizer 触发写入）
        DumpPayload payload = exporter.snapshotPending(false);
        g_pending_json_payload = std::move(payload);
        g_pending_json_dir = dir;
        g_has_pending_json = true;
    }

    // TODO: 在此处调用 ASIC 硬件求解，并填充 result.pose_updates / landmark_updates
    // 当前返回 success=false，外层将回退到原生 BA
    result.success = false;
    return result;
}

void HardwareAdapter::FillExporter(const LocalBAInput& input,
                                   VariableEliminationDriver& driver,
                                   ORBSLAM3Exporter& exporter) {
    // 集合便于查找
    std::set<KeyFrame*> local_set(input.local_kfs.begin(), input.local_kfs.end());
    std::set<KeyFrame*> fixed_set(input.fixed_kfs.begin(), input.fixed_kfs.end());

    // 1) 关键帧初值
    for (KeyFrame* kf : input.local_kfs) {
        Eigen::Matrix3f Rcw = kf->GetRotation();
        Eigen::Vector3f tcw = kf->GetTranslation();
        driver.setPose(kf->mnId, Rcw, tcw);
    }
    for (KeyFrame* kf : input.fixed_kfs) {
        Eigen::Matrix3f Rcw = kf->GetRotation();
        Eigen::Vector3f tcw = kf->GetTranslation();
        driver.setPose(kf->mnId, Rcw, tcw);
    }

    // 2) 路标初值
    if (!input.pose_only) {
        for (MapPoint* mp : input.local_mps) {
            driver.setLandmark(mp->mnId, mp->GetWorldPos().cast<float>());
        }
    }

    // 3) 相机观测因子（位姿-路标因子），若是 pose-only 模式则跳过
    if (!input.pose_only) {
        for (MapPoint* mp : input.local_mps) {
            const auto& obs = mp->GetObservations();
            for (const auto& kv : obs) {
                KeyFrame* kf = kv.first;
                if (!kf || kf->isBad()) continue;
                if (!IsKeyFrameInSet(kf, local_set) && !IsKeyFrameInSet(kf, fixed_set)) continue;

                const int idx = std::get<0>(kv.second);
                if (idx < 0 || idx >= static_cast<int>(kf->mvKeysUn.size())) continue;
                const cv::KeyPoint& kp = kf->mvKeysUn[idx];

                ORBSLAM3Exporter::ObservationInput oi;
                oi.pose_id = kf->mnId;
                oi.landmark_id = mp->mnId;
                oi.pixel = Eigen::Vector2f(kp.pt.x, kp.pt.y);
                oi.Rcw = kf->GetRotation();
                oi.tcw = kf->GetTranslation();
                oi.landmark_w = mp->GetWorldPos().cast<float>();
                oi.fx = kf->fx;
                oi.fy = kf->fy;
                oi.cx = kf->cx;
                oi.cy = kf->cy;
                oi.sigma_pixel = 1.0f;

                exporter.addObservation(oi);
            }
        }
    }

    // 4) 暂不添加 IMU 因子（需按实际 IMU 流水对接）
    // 4) IMU 因子（使用 KeyFrame 内部预积分）；仅当不是纯 pose-only 图
    if (input.inertial && !input.pose_only) {
        ImuPreprocessorGTSAM gtsam_pre;
        for (KeyFrame* kf : input.local_kfs) {
            if (!kf || !kf->mpImuPreintegrated || !kf->mPrevKF) continue;
            KeyFrame* kf_prev = kf->mPrevKF;
            if (kf_prev->GetMap() != kf->GetMap()) continue;

            // 构造原始 IMU 序列
            std::vector<ImuPreprocessorGTSAM::ImuMeasurement> seg;
            const auto* pim = kf->mpImuPreintegrated;
            const auto& meas = pim->GetMeasurements();
            if (meas.size() >= 2) {
                for (size_t i = 1; i < meas.size(); ++i) {
                    double dt = meas[i].t - meas[i-1].t;
                    if (dt <= 0) continue;
                    ImuPreprocessorGTSAM::ImuMeasurement m;
                    m.acc = meas[i-1].a.cast<double>();
                    m.gyro = meas[i-1].w.cast<double>();
                    m.dt = dt;
                    seg.push_back(m);
                }
            }
            if (seg.empty()) continue;

            double dt_total = pim->dT;
            Eigen::Matrix3d Ri_world = kf_prev->GetRotation().cast<double>().transpose(); // 近似用 prev KF 的朝向
            auto out = gtsam_pre.getRelativePose(seg, Ri_world, dt_total);

            ORBSLAM3Exporter::ImuEdgeInput edge;
            edge.pose_i_id = kf_prev->mnId;
            edge.pose_j_id = kf->mnId;
            edge.measurements = std::move(seg);
            edge.dt_total = dt_total;

            exporter.addImuEdge(edge); // sigma/deltaR/deltaT 在 makeImuConstraint 内由 GTSAM 输出

            // 覆盖输出到 driver 以便硬件（直接用 gtsam 输出）
            ImuConstraint c;
            c.pose_i_id = edge.pose_i_id;
            c.pose_j_id = edge.pose_j_id;
            c.deltaR = out.deltaR;
            c.deltaT = out.deltaT;
            c.sigma = out.sigma;
            exporter.addImuConstraintDirect(c);
        }
    }

    // 5) Pose-only 边（Sim3/SE3 → Pose3 Between 近似）
    if (input.pose_only && !input.pose_edges.empty()) {
        for (const auto& e : input.pose_edges) {
            ImuConstraint c;
            c.pose_i_id = e.i;
            c.pose_j_id = e.j;
            c.deltaR = e.R;
            c.deltaT = e.t;
            c.sigma = e.sigma;
            exporter.addImuConstraintDirect(c); // 复用 IMU 因子格式作为 Pose Between
        }
    }
}

void HardwareAdapter::AddPriors(const LocalBAInput& input, ORBSLAM3Exporter& exporter) {
    // 对初始关键帧添加先验，防止尺度漂移
    if (!input.map) return;
    const int init_id = input.map->GetInitKFid();
    for (KeyFrame* kf : input.local_kfs) {
        if (kf->mnId == init_id) {
            ORBSLAM3Exporter::PriorInput p;
            p.pose_id = kf->mnId;
            // KeyFrame::GetRotation/GetTranslation already return Eigen types
            p.Rp = kf->GetRotation();
            p.tp = kf->GetTranslation();
            // 粗略先验
            p.sigma << 0.1f, 0.1f, 0.1f, 0.05f, 0.05f, 0.05f;
            exporter.addPrior(p);
            break;
        }
    }
}

DumpPayload HardwareAdapter::BuildDumpPayload(const LocalBAInput& input) {
    // 若需要自定义 dump，可在这里扩展；当前逻辑已在 exporter.dumpPendingToJson 实现
    return {};
}

}  // namespace ORB_SLAM3


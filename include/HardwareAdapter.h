#ifndef ORB_SLAM3_HARDWAREADAPTER_H
#define ORB_SLAM3_HARDWAREADAPTER_H

#include <list>
#include <map>
#include <string>
#include <vector>
#include <atomic>
#include <Eigen/Dense>
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "config/SolverSwitch.h"
#include "io/JsonDumper.h"

class ORBSLAM3Exporter;
class VariableEliminationDriver;

// 全局缓存：保存待写入的 JSON payload
extern DumpPayload g_pending_json_payload;
extern std::string g_pending_json_dir;
extern std::atomic<bool> g_has_pending_json;

// 异步写入 pending JSON（在 LBA 完成后调用）
void FlushPendingJson();

namespace ORB_SLAM3 {

/**
 * HardwareAdapter: 将 Local BA 的窗口数据转换为硬件/JSON 调用所需格式。
 * 当前实现：
 *  - 收集局部/固定关键帧与路标点
 *  - 生成观察因子（相机）与可选先验
 *  - 按开关导出 JSON
 *  - 预留硬件求解钩子，暂未直接调用硬件，返回 success=false 时上层会回退原生 BA
 */
class HardwareAdapter {
public:
    struct LocalBAInput {
        std::list<KeyFrame*> local_kfs;
        std::list<KeyFrame*> fixed_kfs;
        std::list<MapPoint*> local_mps;
        KeyFrame* current_kf = nullptr;
        Map* map = nullptr;
        bool inertial = false;
        bool pose_only = false;   // true: 仅位姿图（Sim3/SE3），不含路标
        bool fix_scale = true;    // 对应 Sim3 固定尺度时置 true；单目回环可置 false 近似转 SE3
        struct PoseEdge {
            int i = -1;
            int j = -1;
            Eigen::Matrix3f R;    // 相对旋转（SE3）
            Eigen::Vector3f t;    // 相对平移（已除尺度）
            Eigen::Matrix<float,6,1> sigma = Eigen::Matrix<float,6,1>::Constant(1e-2f); // 白化前的 stddev
        };
        std::vector<PoseEdge> pose_edges; // pose-only 场景的边
    };

    struct LocalBAResult {
        bool success = false;
        std::map<KeyFrame*, Sophus::SE3f> pose_updates;
        std::map<MapPoint*, Eigen::Vector3f> landmark_updates;
    };

    static LocalBAResult RunLocalBA(const LocalBAInput& input,
                                    const SolverSwitch& sw,
                                    const std::string& json_dir = "");

private:
    static void FillExporter(const LocalBAInput& input,
                             VariableEliminationDriver& driver,
                             ORBSLAM3Exporter& exporter);
    static void AddPriors(const LocalBAInput& input, ORBSLAM3Exporter& exporter);
    static DumpPayload BuildDumpPayload(const LocalBAInput& input);
};

}  // namespace ORB_SLAM3

#endif  // ORB_SLAM3_HARDWAREADAPTER_H


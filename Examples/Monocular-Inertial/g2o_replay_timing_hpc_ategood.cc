#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "G2oTypes.h"
#include "OptimizableTypes.h"
#include "ImuTypes.h"
#include "CameraModels/Pinhole.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

namespace fs = std::filesystem;
using boost::property_tree::ptree;

namespace {

struct PoseRecord {
    int pose_id = -1;
    double timestamp = 0.0;
    Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
    Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
    Eigen::Vector3f vw = Eigen::Vector3f::Zero();
    Eigen::Vector3f ba = Eigen::Vector3f::Zero();
    Eigen::Vector3f bg = Eigen::Vector3f::Zero();
};

struct ObservationRecord {
    int pose_id = -1;
    int landmark_id = -1;
    float pixel_x = 0.0f;
    float pixel_y = 0.0f;
    Eigen::Vector3f landmark_w = Eigen::Vector3f::Zero();
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float sigma = 1.0f;
    float track_depth = std::numeric_limits<float>::infinity();
};

struct PriorRecord {
    int pose_id = -1;
};

struct ImuEdgeRecord {
    int pose_i_id = -1;
    int pose_j_id = -1;
    bool has_preintegrated = false;
    float dt_total = 0.0f;
    Eigen::Matrix3f dR = Eigen::Matrix3f::Identity();
    Eigen::Vector3f dV = Eigen::Vector3f::Zero();
    Eigen::Vector3f dP = Eigen::Vector3f::Zero();
    Eigen::Matrix3f JRg = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f JVg = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f JVa = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f JPg = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f JPa = Eigen::Matrix3f::Zero();
    Eigen::Vector3f ba0 = Eigen::Vector3f::Zero();
    Eigen::Vector3f bg0 = Eigen::Vector3f::Zero();
    Eigen::Matrix<float,15,15> C = Eigen::Matrix<float,15,15>::Zero();
};

struct CallData {
    int call_id = -1;
    std::vector<PoseRecord> poses;
    std::vector<ObservationRecord> observations;
    std::vector<PriorRecord> priors;
    std::vector<ImuEdgeRecord> imu_edges;
    double lambda = 1.0;
};

struct TimingResult {
    std::string sequence;
    std::string call_name;
    int repeat = 0;
    int call_id = -1;
    std::string mode;
    int poses = 0;
    int fixed_poses = 0;
    int free_poses = 0;
    int landmarks = 0;
    int observations = 0;
    int imu_edges = 0;
    int opt_iters = 0;
    int actual_iters = 0;
    int n_vars = 0;
    std::uint64_t nnz_J = 0;
    double lambda = 0.0;
    double build_ms = 0.0;
    double residual_ms = 0.0;
    double linearize_ms = 0.0;
    double sparse_fill_ms = 0.0;
    double symbolic_ms = 0.0;
    double schur_ms = 0.0;
    double factor_ms = 0.0;
    double linear_solve_ms = 0.0;
    double linear_solution_ms = 0.0;
    double update_ms = 0.0;
    double numeric_ms = 0.0;
    double optimize_ms = 0.0;
    double total_ms = 0.0;
    double init_chi2 = 0.0;
    double final_chi2 = 0.0;
    int active_vertices = 0;
    int active_edges = 0;
    int symbolic_calls = 0;
    int numeric_calls = 0;
    std::string symbolic_policy = "default";
    std::string status = "ok";
    std::string message;
};

std::vector<float> ReadFloatArray(const ptree& node) {
    std::vector<float> out;
    for (const auto& kv : node) out.push_back(kv.second.get_value<float>());
    return out;
}

Eigen::Vector3f ToVector3f(const std::vector<float>& v) {
    if (v.size() < 3) return Eigen::Vector3f::Zero();
    return Eigen::Vector3f(v[0], v[1], v[2]);
}

Eigen::Matrix3f ToMatrix3f(const std::vector<float>& v) {
    Eigen::Matrix3f m = Eigen::Matrix3f::Identity();
    if (v.size() < 9) return m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = v[r * 3 + c];
    return m;
}

Eigen::Matrix<float,15,15> ToMatrix15f(const std::vector<float>& v) {
    Eigen::Matrix<float,15,15> m = Eigen::Matrix<float,15,15>::Zero();
    if (v.size() < 225) return m;
    for (int r = 0; r < 15; ++r)
        for (int c = 0; c < 15; ++c)
            m(r, c) = v[r * 15 + c];
    return m;
}

std::vector<PoseRecord> LoadPoses(const fs::path& file) {
    ptree root;
    boost::property_tree::read_json(file.string(), root);
    std::vector<PoseRecord> poses;
    for (const auto& entry : root.get_child("poses")) {
        PoseRecord p;
        p.pose_id = entry.second.get<int>("pose_id");
        p.timestamp = entry.second.get<double>("timestamp", 0.0);
        p.Rcw = ToMatrix3f(ReadFloatArray(entry.second.get_child("Rcw")));
        p.tcw = ToVector3f(ReadFloatArray(entry.second.get_child("tcw")));
        if (auto n = entry.second.get_child_optional("vw")) p.vw = ToVector3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("ba")) p.ba = ToVector3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("bg")) p.bg = ToVector3f(ReadFloatArray(*n));
        poses.push_back(p);
    }
    return poses;
}

std::vector<ObservationRecord> LoadObservations(const fs::path& file) {
    ptree root;
    boost::property_tree::read_json(file.string(), root);
    std::vector<ObservationRecord> observations;
    for (const auto& entry : root.get_child("observations")) {
        ObservationRecord o;
        o.pose_id = entry.second.get<int>("pose_id");
        o.landmark_id = entry.second.get<int>("landmark_id");
        auto pixel = ReadFloatArray(entry.second.get_child("pixel"));
        if (pixel.size() >= 2) {
            o.pixel_x = pixel[0];
            o.pixel_y = pixel[1];
        }
        o.landmark_w = ToVector3f(ReadFloatArray(entry.second.get_child("landmark_w")));
        auto intr = ReadFloatArray(entry.second.get_child("intrinsics"));
        if (intr.size() >= 4) {
            o.fx = intr[0];
            o.fy = intr[1];
            o.cx = intr[2];
            o.cy = intr[3];
        }
        o.sigma = entry.second.get<float>("sigma_pixel", 1.0f);
        if (o.sigma <= 0.0f) o.sigma = 1.0f;
        o.track_depth = entry.second.get<float>("track_depth", std::numeric_limits<float>::infinity());
        observations.push_back(o);
    }
    return observations;
}

std::vector<PriorRecord> LoadPriors(const fs::path& file) {
    ptree root;
    boost::property_tree::read_json(file.string(), root);
    std::vector<PriorRecord> priors;
    for (const auto& entry : root.get_child("priors")) {
        PriorRecord p;
        p.pose_id = entry.second.get<int>("pose_id");
        priors.push_back(p);
    }
    return priors;
}

std::vector<ImuEdgeRecord> LoadImuEdges(const fs::path& file) {
    ptree root;
    boost::property_tree::read_json(file.string(), root);
    std::vector<ImuEdgeRecord> edges;
    for (const auto& entry : root.get_child("imu_edges")) {
        ImuEdgeRecord e;
        e.pose_i_id = entry.second.get<int>("pose_i");
        e.pose_j_id = entry.second.get<int>("pose_j");
        if (auto n = entry.second.get_child_optional("dt_total")) e.dt_total = n->get_value<float>();
        if (auto n = entry.second.get_child_optional("dR")) e.dR = ToMatrix3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("dV")) e.dV = ToVector3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("dP")) e.dP = ToVector3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("JRg")) e.JRg = ToMatrix3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("JVg")) e.JVg = ToMatrix3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("JVa")) e.JVa = ToMatrix3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("JPg")) e.JPg = ToMatrix3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("JPa")) e.JPa = ToMatrix3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("ba0")) e.ba0 = ToVector3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("bg0")) e.bg0 = ToVector3f(ReadFloatArray(*n));
        if (auto n = entry.second.get_child_optional("C")) e.C = ToMatrix15f(ReadFloatArray(*n));
        e.has_preintegrated = entry.second.get_child_optional("dR") &&
                              entry.second.get_child_optional("dV") &&
                              entry.second.get_child_optional("dP") &&
                              e.dt_total > 0.0f;
        edges.push_back(e);
    }
    return edges;
}

int ParseCallId(const fs::path& call_dir) {
    const std::string name = call_dir.filename().string();
    for (const std::string& prefix : {std::string("call_"), std::string("lba_kf_")}) {
        if (name.rfind(prefix, 0) == 0) return std::stoi(name.substr(prefix.size()));
    }
    return -1;
}

double ParseClosureLambda(const fs::path& call_dir) {
    std::ifstream in(call_dir / "CLOSURE.txt");
    if (!in) return 1.0;
    std::string line;
    std::regex re("stim_lambda=([0-9eE+\\-.]+)");
    while (std::getline(in, line)) {
        std::smatch m;
        if (std::regex_search(line, m, re)) return std::stod(m[1]);
    }
    return 1.0;
}

CallData LoadCall(const fs::path& call_dir) {
    const fs::path filtered = call_dir / "inputs_filtered";
    const fs::path inputs = fs::is_directory(filtered) ? filtered : call_dir;
    CallData data;
    data.call_id = ParseCallId(call_dir);
    data.poses = LoadPoses(inputs / "poses.json");
    data.observations = LoadObservations(inputs / "camera_observations.json");
    data.priors = LoadPriors(inputs / "priors.json");
    data.imu_edges = LoadImuEdges(inputs / "imu_edges.json");
    data.lambda = ParseClosureLambda(call_dir);
    return data;
}

std::set<int> FixedPoses(const CallData& data) {
    std::set<int> fixed;
    for (const auto& p : data.priors) fixed.insert(p.pose_id);
    return fixed;
}

std::map<int, PoseRecord> PoseMap(const CallData& data) {
    std::map<int, PoseRecord> poses;
    for (const auto& p : data.poses) poses[p.pose_id] = p;
    return poses;
}

std::map<int, Eigen::Vector3f> LandmarkMap(const CallData& data) {
    std::map<int, Eigen::Vector3f> lms;
    for (const auto& o : data.observations) {
        if (!lms.count(o.landmark_id)) lms[o.landmark_id] = o.landmark_w;
    }
    return lms;
}

int MaxPoseId(const CallData& data) {
    int max_id = 0;
    for (const auto& p : data.poses) max_id = std::max(max_id, p.pose_id);
    return max_id;
}

std::vector<float> IntrinsicsForPose(const CallData& data, int pose_id) {
    for (const auto& o : data.observations) {
        if (o.pose_id == pose_id) return {o.fx, o.fy, o.cx, o.cy};
    }
    if (!data.observations.empty()) {
        const auto& o = data.observations.front();
        return {o.fx, o.fy, o.cx, o.cy};
    }
    return {458.654f, 457.296f, 367.215f, 248.375f};
}

ORB_SLAM3::ImuCamPose MakeImuCamPose(const PoseRecord& p, ORB_SLAM3::GeometricCamera* cam) {
    Eigen::Matrix3d Rcw = p.Rcw.cast<double>();
    Eigen::Vector3d tcw = p.tcw.cast<double>();

    Eigen::Matrix3d Rbc;
    Rbc << 0.014865543, -0.999880910,  0.004140297,
           0.999557257,  0.014967213,  0.025715530,
          -0.025774436,  0.003756188,  0.999660730;
    Eigen::Vector3d tbc(-0.021640146, -0.064676985, 0.009810731);

    ORB_SLAM3::ImuCamPose est;
    est.SetParam({Rcw}, {tcw}, {Rbc}, {tbc}, 0.0);
    est.pCamera.resize(1);
    est.pCamera[0] = cam;
    est.Rwb0 = est.Rwb;
    est.DR.setIdentity();
    est.its = 0;
    return est;
}

Eigen::Matrix3d SafeInverseInfo3(const Eigen::Matrix3f& Cf) {
    Eigen::Matrix3d C = Cf.cast<double>();
    C = 0.5 * (C + C.transpose());
    if (!C.allFinite() || C.cwiseAbs().maxCoeff() <= 0.0) {
        return Eigen::Matrix3d::Identity() * 1e12;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
    Eigen::Vector3d inv;
    for (int i = 0; i < 3; ++i) inv(i) = 1.0 / std::max(es.eigenvalues()(i), 1e-18);
    return es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
}

template <typename OptimizerPtr>
void SetLevenbergUserLambda(OptimizerPtr* algorithm, double lambda) {
    if (lambda > 0.0 && std::isfinite(lambda)) algorithm->setUserLambdaInit(lambda);
}

enum class SymbolicPolicy {
    kDefault,
    kForcePerCall,
    kEverySolve,
};

const char* SymbolicPolicyName(SymbolicPolicy policy) {
    switch (policy) {
        case SymbolicPolicy::kDefault: return "default";
        case SymbolicPolicy::kForcePerCall: return "force_per_call";
        case SymbolicPolicy::kEverySolve: return "every_solve";
    }
    return "unknown";
}

struct LinearSolverTiming {
    double sparse_fill_ms = 0.0;
    double symbolic_ms = 0.0;
    double factor_ms = 0.0;
    double solve_ms = 0.0;
    int symbolic_calls = 0;
    int numeric_calls = 0;
};

template <typename MatrixType>
class InstrumentedLinearSolverEigen : public g2o::LinearSolverEigen<MatrixType> {
public:
    explicit InstrumentedLinearSolverEigen(SymbolicPolicy policy) : policy_(policy) {}

    bool solve(const g2o::SparseBlockMatrix<MatrixType>& A,
               g2o::number_t* x,
               g2o::number_t* b) override {
        const bool was_init = this->_init;
        if (was_init) this->_sparseMatrix.resize(A.rows(), A.cols());

        auto t0 = std::chrono::steady_clock::now();
        this->fillSparseMatrix(A, !was_init);
        auto t1 = std::chrono::steady_clock::now();
        timing_.sparse_fill_ms += Milliseconds(t1 - t0);

        if (was_init || policy_ == SymbolicPolicy::kEverySolve) {
            t0 = std::chrono::steady_clock::now();
            this->computeSymbolicDecomposition(A);
            t1 = std::chrono::steady_clock::now();
            timing_.symbolic_ms += Milliseconds(t1 - t0);
            ++timing_.symbolic_calls;
        }
        this->_init = false;

        t0 = std::chrono::steady_clock::now();
        this->_cholesky.factorize(this->_sparseMatrix);
        t1 = std::chrono::steady_clock::now();
        const double factor_ms = Milliseconds(t1 - t0);
        timing_.factor_ms += factor_ms;
        if (this->_cholesky.info() != Eigen::Success) {
            if (this->_writeDebug) A.writeOctave("debug.txt");
            return false;
        }

        g2o::VectorXD::MapType xx(x, this->_sparseMatrix.cols());
        g2o::VectorXD::ConstMapType bb(b, this->_sparseMatrix.cols());
        t0 = std::chrono::steady_clock::now();
        xx = this->_cholesky.solve(bb);
        t1 = std::chrono::steady_clock::now();
        const double solve_ms = Milliseconds(t1 - t0);
        timing_.solve_ms += solve_ms;
        ++timing_.numeric_calls;

        g2o::G2OBatchStatistics* stats = g2o::G2OBatchStatistics::globalStats();
        if (stats) {
            stats->timeNumericDecomposition = (factor_ms + solve_ms) * 1e-3;
            stats->choleskyNNZ = this->_cholesky.matrixL().nestedExpression().nonZeros() +
                                 this->_sparseMatrix.cols();
        }
        return true;
    }

    const LinearSolverTiming& timing() const { return timing_; }

private:
    template <typename Duration>
    static double Milliseconds(const Duration& duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    SymbolicPolicy policy_;
    LinearSolverTiming timing_;
};

void CollectBatchTimings(const g2o::SparseOptimizer& optimizer, TimingResult& result) {
    for (const auto& stats : optimizer.batchStatistics()) {
        result.residual_ms += 1000.0 * stats.timeResiduals;
        result.linearize_ms += 1000.0 * stats.timeQuadraticForm;
        result.schur_ms += 1000.0 * stats.timeSchurComplement;
        result.linear_solution_ms += 1000.0 * stats.timeLinearSolution;
        result.update_ms += 1000.0 * stats.timeUpdate;
    }
}

void CollectLinearSolverTimings(const LinearSolverTiming& timing, TimingResult& result) {
    result.sparse_fill_ms = timing.sparse_fill_ms;
    result.symbolic_ms = timing.symbolic_ms;
    result.factor_ms = timing.factor_ms;
    result.linear_solve_ms = timing.solve_ms;
    result.numeric_ms = std::max(0.0, result.linear_solution_ms - timing.symbolic_ms) +
                        result.update_ms;
    result.symbolic_calls = timing.symbolic_calls;
    result.numeric_calls = timing.numeric_calls;
}

TimingResult RunVisualG2O(const CallData& data, int opt_iters, SymbolicPolicy policy) {
    TimingResult result;
    result.call_id = data.call_id;
    result.mode = "visual";
    result.poses = static_cast<int>(data.poses.size());
    result.observations = static_cast<int>(data.observations.size());
    result.imu_edges = static_cast<int>(data.imu_edges.size());
    result.lambda = data.lambda;
    result.opt_iters = opt_iters;

    const auto fixed = FixedPoses(data);
    const auto landmarks = LandmarkMap(data);
    result.fixed_poses = static_cast<int>(fixed.size());
    result.free_poses = result.poses - result.fixed_poses;
    result.landmarks = static_cast<int>(landmarks.size());
    result.n_vars = result.free_poses + result.landmarks;
    result.nnz_J = static_cast<std::uint64_t>(result.observations) * 18;
    result.symbolic_policy = SymbolicPolicyName(policy);

    std::vector<std::unique_ptr<ORB_SLAM3::Pinhole>> cameras;

    const auto t_build0 = std::chrono::steady_clock::now();

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto* linear_solver =
        new InstrumentedLinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>(policy);
    auto* block_solver = new g2o::BlockSolver_6_3(linear_solver);
    auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    SetLevenbergUserLambda(algorithm, data.lambda);
    optimizer.setAlgorithm(algorithm);

    for (const auto& p : data.poses) {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(p.pose_id);
        v->setFixed(fixed.count(p.pose_id));
        v->setEstimate(g2o::SE3Quat(p.Rcw.cast<double>(), p.tcw.cast<double>()));
        optimizer.addVertex(v);
    }

    const int max_pose_id = MaxPoseId(data);
    const int landmark_base = max_pose_id + 1;
    for (const auto& kv : landmarks) {
        auto* v = new g2o::VertexSBAPointXYZ();
        v->setId(landmark_base + kv.first);
        v->setMarginalized(true);
        v->setEstimate(kv.second.cast<double>());
        optimizer.addVertex(v);
    }

    std::map<std::vector<float>, ORB_SLAM3::Pinhole*> camera_cache;
    auto get_camera = [&](const ObservationRecord& o) -> ORB_SLAM3::Pinhole* {
        std::vector<float> key{o.fx, o.fy, o.cx, o.cy};
        auto it = camera_cache.find(key);
        if (it != camera_cache.end()) return it->second;
        cameras.emplace_back(std::make_unique<ORB_SLAM3::Pinhole>(key));
        ORB_SLAM3::Pinhole* cam = cameras.back().get();
        camera_cache[key] = cam;
        return cam;
    };

    for (const auto& o : data.observations) {
        auto* e = new ORB_SLAM3::EdgeSE3ProjectXYZ();
        e->setVertex(0, optimizer.vertex(landmark_base + o.landmark_id));
        e->setVertex(1, optimizer.vertex(o.pose_id));
        e->setMeasurement(Eigen::Vector2d(o.pixel_x, o.pixel_y));
        const double inv_sigma2 = 1.0 / std::max(1e-12, static_cast<double>(o.sigma) * static_cast<double>(o.sigma));
        e->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);
        auto* rk = new g2o::RobustKernelHuber;
        rk->setDelta(std::sqrt(5.991));
        e->setRobustKernel(rk);
        e->pCamera = get_camera(o);
        optimizer.addEdge(e);
    }

    const auto t_build1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.setComputeBatchStatistics(true);
    optimizer.computeActiveErrors();
    result.init_chi2 = optimizer.activeRobustChi2();

    const auto t_opt0 = std::chrono::steady_clock::now();
    result.actual_iters = optimizer.optimize(opt_iters);
    const auto t_opt1 = std::chrono::steady_clock::now();
    optimizer.computeActiveErrors();
    result.final_chi2 = optimizer.activeRobustChi2();

    result.build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();
    result.optimize_ms = std::chrono::duration<double, std::milli>(t_opt1 - t_opt0).count();
    result.total_ms = std::chrono::duration<double, std::milli>(t_opt1 - t_build0).count();
    result.active_vertices = static_cast<int>(optimizer.vertices().size());
    result.active_edges = static_cast<int>(optimizer.edges().size());
    CollectBatchTimings(optimizer, result);
    CollectLinearSolverTimings(linear_solver->timing(), result);
    return result;
}

TimingResult RunInertialG2O(const CallData& data, int opt_iters, SymbolicPolicy policy) {
    TimingResult result;
    result.call_id = data.call_id;
    result.mode = "inertial";
    result.poses = static_cast<int>(data.poses.size());
    result.observations = static_cast<int>(data.observations.size());
    result.imu_edges = static_cast<int>(data.imu_edges.size());
    result.lambda = data.lambda;
    result.opt_iters = opt_iters;

    const auto fixed = FixedPoses(data);
    const auto pose_map = PoseMap(data);
    const auto landmarks = LandmarkMap(data);
    result.fixed_poses = static_cast<int>(fixed.size());
    result.free_poses = result.poses - result.fixed_poses;
    result.landmarks = static_cast<int>(landmarks.size());
    result.n_vars = 4 * result.free_poses + result.landmarks;
    result.nnz_J = static_cast<std::uint64_t>(result.observations) * 18 +
                   static_cast<std::uint64_t>(result.imu_edges) * 252;
    result.symbolic_policy = SymbolicPolicyName(policy);

    std::vector<std::unique_ptr<ORB_SLAM3::Pinhole>> cameras;
    std::vector<std::unique_ptr<ORB_SLAM3::IMU::Preintegrated>> preintegrations;

    const auto t_build0 = std::chrono::steady_clock::now();

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    auto* linear_solver =
        new InstrumentedLinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>(policy);
    auto* block_solver = new g2o::BlockSolverX(linear_solver);
    auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
    SetLevenbergUserLambda(algorithm, data.lambda);
    optimizer.setAlgorithm(algorithm);

    const int max_pose_id = MaxPoseId(data);
    auto vertex_velocity_id = [&](int pose_id) { return max_pose_id + 3 * pose_id + 1; };
    auto vertex_gyro_id = [&](int pose_id) { return max_pose_id + 3 * pose_id + 2; };
    auto vertex_acc_id = [&](int pose_id) { return max_pose_id + 3 * pose_id + 3; };

    for (const auto& p : data.poses) {
        const std::vector<float> intr = IntrinsicsForPose(data, p.pose_id);
        cameras.emplace_back(std::make_unique<ORB_SLAM3::Pinhole>(intr));
        ORB_SLAM3::Pinhole* cam = cameras.back().get();
        const bool is_fixed = fixed.count(p.pose_id) > 0;

        auto* vp = new ORB_SLAM3::VertexPose();
        vp->setId(p.pose_id);
        vp->setFixed(is_fixed);
        vp->setEstimate(MakeImuCamPose(p, cam));
        optimizer.addVertex(vp);

        auto* vv = new ORB_SLAM3::VertexVelocity();
        vv->setId(vertex_velocity_id(p.pose_id));
        vv->setFixed(is_fixed);
        vv->setEstimate(p.vw.cast<double>());
        optimizer.addVertex(vv);

        auto* vg = new ORB_SLAM3::VertexGyroBias();
        vg->setId(vertex_gyro_id(p.pose_id));
        vg->setFixed(is_fixed);
        vg->setEstimate(p.bg.cast<double>());
        optimizer.addVertex(vg);

        auto* va = new ORB_SLAM3::VertexAccBias();
        va->setId(vertex_acc_id(p.pose_id));
        va->setFixed(is_fixed);
        va->setEstimate(p.ba.cast<double>());
        optimizer.addVertex(va);
    }

    // Pose p occupies IDs p, max_pose+3p+1, +2, and +3. Start landmarks
    // strictly above the last possible inertial state ID, including tiny graphs.
    const int landmark_base = max_pose_id * 4 + 4;
    for (const auto& kv : landmarks) {
        auto* v = new g2o::VertexSBAPointXYZ();
        v->setId(landmark_base + kv.first);
        v->setMarginalized(true);
        v->setEstimate(kv.second.cast<double>());
        optimizer.addVertex(v);
    }

    for (const auto& e_in : data.imu_edges) {
        if (!e_in.has_preintegrated) continue;
        auto pi_it = pose_map.find(e_in.pose_i_id);
        auto pj_it = pose_map.find(e_in.pose_j_id);
        if (pi_it == pose_map.end() || pj_it == pose_map.end()) continue;

        auto preint = std::make_unique<ORB_SLAM3::IMU::Preintegrated>();
        preint->dT = e_in.dt_total;
        preint->C = e_in.C;
        preint->dR = e_in.dR;
        preint->dV = e_in.dV;
        preint->dP = e_in.dP;
        preint->JRg = e_in.JRg;
        preint->JVg = e_in.JVg;
        preint->JVa = e_in.JVa;
        preint->JPg = e_in.JPg;
        preint->JPa = e_in.JPa;
        preint->b = ORB_SLAM3::IMU::Bias(e_in.ba0.x(), e_in.ba0.y(), e_in.ba0.z(),
                                          e_in.bg0.x(), e_in.bg0.y(), e_in.bg0.z());
        const auto& pi_pose = pi_it->second;
        preint->SetNewBias(ORB_SLAM3::IMU::Bias(pi_pose.ba.x(), pi_pose.ba.y(), pi_pose.ba.z(),
                                                pi_pose.bg.x(), pi_pose.bg.y(), pi_pose.bg.z()));

        auto* edge = new ORB_SLAM3::EdgeInertial(preint.get());
        edge->setVertex(0, optimizer.vertex(e_in.pose_i_id));
        edge->setVertex(1, optimizer.vertex(vertex_velocity_id(e_in.pose_i_id)));
        edge->setVertex(2, optimizer.vertex(vertex_gyro_id(e_in.pose_i_id)));
        edge->setVertex(3, optimizer.vertex(vertex_acc_id(e_in.pose_i_id)));
        edge->setVertex(4, optimizer.vertex(e_in.pose_j_id));
        edge->setVertex(5, optimizer.vertex(vertex_velocity_id(e_in.pose_j_id)));
        const bool boundary = fixed.count(e_in.pose_i_id) && !fixed.count(e_in.pose_j_id);
        if (boundary) {
            auto* rk = new g2o::RobustKernelHuber;
            rk->setDelta(std::sqrt(16.92));
            edge->setRobustKernel(rk);
            edge->setInformation(edge->information() * 1e-2);
        }
        optimizer.addEdge(edge);

        auto* egr = new ORB_SLAM3::EdgeGyroRW();
        egr->setVertex(0, optimizer.vertex(vertex_gyro_id(e_in.pose_i_id)));
        egr->setVertex(1, optimizer.vertex(vertex_gyro_id(e_in.pose_j_id)));
        egr->setInformation(SafeInverseInfo3(e_in.C.block<3,3>(9,9)).cast<g2o::number_t>());
        optimizer.addEdge(egr);

        auto* ear = new ORB_SLAM3::EdgeAccRW();
        ear->setVertex(0, optimizer.vertex(vertex_acc_id(e_in.pose_i_id)));
        ear->setVertex(1, optimizer.vertex(vertex_acc_id(e_in.pose_j_id)));
        ear->setInformation(SafeInverseInfo3(e_in.C.block<3,3>(12,12)).cast<g2o::number_t>());
        optimizer.addEdge(ear);

        preintegrations.emplace_back(std::move(preint));
    }

    for (const auto& o : data.observations) {
        auto* e = new ORB_SLAM3::EdgeMono(0);
        e->setVertex(0, optimizer.vertex(landmark_base + o.landmark_id));
        e->setVertex(1, optimizer.vertex(o.pose_id));
        e->setMeasurement(Eigen::Vector2d(o.pixel_x, o.pixel_y));
        const double inv_sigma2 = 1.0 / std::max(1e-12, static_cast<double>(o.sigma) * static_cast<double>(o.sigma));
        e->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);
        auto* rk = new g2o::RobustKernelHuber;
        rk->setDelta(std::sqrt(5.991));
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
    }

    const auto t_build1 = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.setComputeBatchStatistics(true);
    optimizer.computeActiveErrors();
    result.init_chi2 = optimizer.activeRobustChi2();

    const auto t_opt0 = std::chrono::steady_clock::now();
    result.actual_iters = optimizer.optimize(opt_iters);
    const auto t_opt1 = std::chrono::steady_clock::now();
    optimizer.computeActiveErrors();
    result.final_chi2 = optimizer.activeRobustChi2();

    result.build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();
    result.optimize_ms = std::chrono::duration<double, std::milli>(t_opt1 - t_opt0).count();
    result.total_ms = std::chrono::duration<double, std::milli>(t_opt1 - t_build0).count();
    result.active_vertices = static_cast<int>(optimizer.vertices().size());
    result.active_edges = static_cast<int>(optimizer.edges().size());
    CollectBatchTimings(optimizer, result);
    CollectLinearSolverTimings(linear_solver->timing(), result);
    return result;
}

TimingResult RunCall(const CallData& data, int opt_iters, SymbolicPolicy policy) {
    TimingResult result;
    try {
        if (data.imu_edges.empty()) result = RunVisualG2O(data, opt_iters, policy);
        else result = RunInertialG2O(data, opt_iters, policy);
    } catch (const std::exception& e) {
        result.call_id = data.call_id;
        result.status = "error";
        result.message = e.what();
    }
    return result;
}

std::string CsvEscape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

void WriteHeader(std::ostream& os) {
    os << "sequence,call_id,call_name,repeat,mode,poses,fixed_poses,free_poses,"
          "landmarks,observations,imu_edges,n_vars,nnz_J,opt_iters,actual_iters,lambda,"
          "build_ms,T_residual_ms,T_lin_ms,T_sparse_fill_ms,T_sym_ms,T_factor_ms,"
          "T_schur_ms,T_solve_ms,T_linear_solution_ms,T_update_ms,T_num_ms,"
          "optimize_ms,total_ms,sym_calls,numeric_calls,"
          "symbolic_policy,init_chi2,final_chi2,active_vertices,active_edges,status,message\n";
}

void WriteResult(std::ostream& os, const TimingResult& r) {
    os << CsvEscape(r.sequence) << ',' << r.call_id << ',' << CsvEscape(r.call_name) << ','
       << r.repeat << ',' << r.mode << ',' << r.poses << ',' << r.fixed_poses << ','
       << r.free_poses << ',' << r.landmarks << ',' << r.observations << ',' << r.imu_edges << ','
       << r.n_vars << ',' << r.nnz_J << ',' << r.opt_iters << ',' << r.actual_iters << ','
       << std::setprecision(12) << r.lambda << ',' << r.build_ms << ',' << r.residual_ms << ','
       << r.linearize_ms << ',' << r.sparse_fill_ms << ',' << r.symbolic_ms << ','
       << r.factor_ms << ',' << r.schur_ms << ',' << r.linear_solve_ms << ','
       << r.linear_solution_ms << ',' << r.update_ms << ',' << r.numeric_ms << ','
       << r.optimize_ms << ',' << r.total_ms << ',' << r.symbolic_calls << ','
       << r.numeric_calls << ',' << r.symbolic_policy << ',' << r.init_chi2 << ','
       << r.final_chi2 << ',' << r.active_vertices << ',' << r.active_edges << ','
       << r.status << ',' << CsvEscape(r.message) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <call_root> <output_csv> [--iters=N] [--repeats=N] [--warmup=N]"
                     " [--sequence=NAME]"
                     " [--symbolic-policy=default|force-per-call|every-solve]"
                     " [--single-call=N] [--limit=N]\n";
        return 1;
    }
    fs::path fullseq_dir = argv[1];
    fs::path output_csv = argv[2];
    int opt_iters = 1;
    int repeats = 1;
    int warmup = 0;
    int single_call = -1;
    int limit = -1;
    std::string sequence = fullseq_dir.filename().string();
    SymbolicPolicy symbolic_policy = SymbolicPolicy::kDefault;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--iters=", 0) == 0) opt_iters = std::max(1, std::stoi(arg.substr(8)));
        else if (arg.rfind("--repeats=", 0) == 0) repeats = std::max(1, std::stoi(arg.substr(10)));
        else if (arg.rfind("--warmup=", 0) == 0) warmup = std::max(0, std::stoi(arg.substr(9)));
        else if (arg.rfind("--sequence=", 0) == 0) sequence = arg.substr(11);
        else if (arg == "--symbolic-policy=default") symbolic_policy = SymbolicPolicy::kDefault;
        else if (arg == "--symbolic-policy=force-per-call") symbolic_policy = SymbolicPolicy::kForcePerCall;
        else if (arg == "--symbolic-policy=every-solve") symbolic_policy = SymbolicPolicy::kEverySolve;
        else if (arg.rfind("--single-call=", 0) == 0) single_call = std::stoi(arg.substr(14));
        else if (arg.rfind("--limit=", 0) == 0) limit = std::stoi(arg.substr(8));
        else {
            std::cerr << "Unknown arg: " << arg << "\n";
            return 1;
        }
    }

    std::vector<fs::path> call_dirs;
    for (const auto& entry : fs::directory_iterator(fullseq_dir)) {
        if (!entry.is_directory()) continue;
        const int call_id = ParseCallId(entry.path());
        if (call_id < 0) continue;
        if (single_call >= 0 && call_id != single_call) continue;
        call_dirs.push_back(entry.path());
    }
    std::sort(call_dirs.begin(), call_dirs.end(), [](const fs::path& a, const fs::path& b) {
        return ParseCallId(a) < ParseCallId(b);
    });
    if (limit >= 0 && static_cast<int>(call_dirs.size()) > limit) call_dirs.resize(limit);

    std::ofstream out(output_csv);
    if (!out) {
        std::cerr << "Cannot open output CSV: " << output_csv << "\n";
        return 1;
    }
    out << std::fixed << std::setprecision(6);
    WriteHeader(out);

    int ok = 0;
    int err = 0;
    for (const auto& d : call_dirs) {
        CallData data;
        try {
            data = LoadCall(d);
        } catch (const std::exception& e) {
            TimingResult load_error;
            load_error.sequence = sequence;
            load_error.call_id = ParseCallId(d);
            load_error.call_name = d.filename().string();
            load_error.status = "error";
            load_error.message = e.what();
            WriteResult(out, load_error);
            out.flush();
            ++err;
            std::cerr << "[G2O_REPLAY] sequence=" << sequence << " call="
                      << load_error.call_id << " status=error msg=" << e.what() << "\n";
            continue;
        }

        for (int i = 0; i < warmup; ++i) {
            TimingResult ignored = RunCall(data, opt_iters, symbolic_policy);
            if (ignored.status != "ok") break;
        }

        bool call_ok = true;
        TimingResult last;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            TimingResult r = RunCall(data, opt_iters, symbolic_policy);
            r.sequence = sequence;
            r.call_name = d.filename().string();
            r.repeat = repeat;
            WriteResult(out, r);
            out.flush();
            if (r.status != "ok") call_ok = false;
            last = std::move(r);
        }
        if (call_ok) ++ok;
        else ++err;
        std::cerr << "[G2O_REPLAY] sequence=" << sequence << " call=" << last.call_id
                  << " mode=" << last.mode << " poses=" << last.poses
                  << " fixed=" << last.fixed_poses << " lms=" << last.landmarks
                  << " obs=" << last.observations << " imu=" << last.imu_edges
                  << " lin_ms=" << last.linearize_ms << " sym_ms=" << last.symbolic_ms
                  << " num_ms=" << last.numeric_ms << " opt_ms=" << last.optimize_ms
                  << " repeats=" << repeats << " status=" << last.status;
        if (!last.message.empty()) std::cerr << " msg=" << last.message;
        std::cerr << "\n";
    }
    std::cerr << "[G2O_REPLAY_DONE] calls=" << call_dirs.size() << " ok=" << ok << " error=" << err
              << " iters=" << opt_iters << " repeats=" << repeats << " warmup=" << warmup
              << " csv=" << output_csv << "\n";
    return err == 0 ? 0 : 2;
}

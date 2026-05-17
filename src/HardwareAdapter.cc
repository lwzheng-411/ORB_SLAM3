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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

#ifndef LANDMARK_CAP
#define LANDMARK_CAP 2147483647
#endif
#ifndef ASIC_FEATURES_PER_KF_CAP
#define ASIC_FEATURES_PER_KF_CAP 2147483647
#endif
#ifndef ASIC_OBS_PER_LANDMARK_CAP
#define ASIC_OBS_PER_LANDMARK_CAP 2147483647
#endif

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

std::string CpuQrSolverPath() {
    const char* env = std::getenv("CPU_QR_SOLVER_BIN");
    if (env && env[0] != '\0') return std::string(env);
    return "/home/zlw/End2End/QR/SW/json/measure_cpu_qr";
}

std::string CpuQrScratchRoot() {
    const char* env = std::getenv("CPU_QR_SCRATCH_DIR");
    if (env && env[0] != '\0') return std::string(env);
    return "/tmp/orbslam3_cpu_qr_online";
}

// =================================================================
// Persistent solver IPC. Activated by env CPU_QR_USE_PERSISTENT_SOLVER=1.
// Protocol (lines, \n-terminated):
//   request : "SOLVE\t<json_dir>\t<output_path>\t<extra args sep by space>"
//   response: "DONE\t<rc>"
//   close   : "QUIT" (driver -> server) -> server exits cleanly
// On any failure (write/read/EOF/parse), the driver tears down state and
// signals the caller to fall back to fork+exec for that call. The next call
// will respawn the server.
// =================================================================
struct PersistentSolverState {
    std::mutex mtx;
    pid_t pid = -1;
    int   fd  = -1;       // socketpair fd on parent side (read+write)
    bool  initialized = false;
    std::string bin_path;
};

PersistentSolverState& GetPersistentSolverState() {
    static PersistentSolverState s;
    return s;
}

void TeardownPersistentSolver_NoLock(PersistentSolverState& s, const char* reason) {
    if (s.fd >= 0) { close(s.fd); s.fd = -1; }
    if (s.pid > 0) {
        // Best-effort: send SIGTERM, reap. Don't block the LBA path.
        kill(s.pid, SIGTERM);
        int status = 0;
        for (int i = 0; i < 20; ++i) {  // ~200ms wait
            pid_t r = waitpid(s.pid, &status, WNOHANG);
            if (r == s.pid || r < 0) break;
            usleep(10 * 1000);
        }
        // If still alive, SIGKILL.
        if (waitpid(s.pid, &status, WNOHANG) == 0) {
            kill(s.pid, SIGKILL);
            waitpid(s.pid, &status, 0);
        }
        s.pid = -1;
    }
    s.initialized = false;
    if (reason) std::cerr << "[CPU_QR_PERSIST] tore down (" << reason << ")\n";
}

// Caller must hold s.mtx. Returns true if server is up and reachable.
bool EnsurePersistentSolver_NoLock(PersistentSolverState& s) {
    if (s.initialized && s.fd >= 0 && s.pid > 0) return true;
    // Build solver binary path: prefer CPU_QR_SOLVER_BIN, else server-named path.
    std::string bin = CpuQrSolverPath();
    s.bin_path = bin;
    int sv[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        std::cerr << "[CPU_QR_PERSIST] socketpair failed: " << std::strerror(errno) << "\n";
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[CPU_QR_PERSIST] fork failed: " << std::strerror(errno) << "\n";
        close(sv[0]); close(sv[1]);
        return false;
    }
    if (pid == 0) {
        // Child: stdin+stdout = sv[1]; stderr stays attached to console.
        if (dup2(sv[1], STDIN_FILENO) < 0)  _exit(120);
        if (dup2(sv[1], STDOUT_FILENO) < 0) _exit(121);
        close(sv[0]); close(sv[1]);
        setenv("CPU_QR_PERSISTENT_MODE", "1", 1);
        // exec the server binary. argv[0] is informative only.
        const char* argv0 = bin.c_str();
        execl(argv0, argv0, (char*)nullptr);
        std::cerr << "[CPU_QR_PERSIST] execl failed: " << std::strerror(errno) << "\n";
        _exit(127);
    }
    // Parent.
    close(sv[1]);
    s.fd = sv[0];
    s.pid = pid;
    s.initialized = true;
    // Make sure SIGPIPE doesn't kill us if the child dies mid-write.
    static std::once_flag sigpipe_once;
    std::call_once(sigpipe_once, []() {
        struct sigaction sa{};
        sa.sa_handler = SIG_IGN;
        sigaction(SIGPIPE, &sa, nullptr);
    });
    std::cerr << "[CPU_QR_PERSIST] server up: pid=" << pid << " bin=" << bin << "\n";
    return true;
}

// Write all bytes; returns true on success.
bool WriteAll(int fd, const char* data, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t r = ::write(fd, data + off, n - off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;
        off += static_cast<size_t>(r);
    }
    return true;
}

// Read until we see a '\n'. Returns the line without trailing '\n'.
// Returns false on EOF / error before any data, OR on partial data with EOF.
bool ReadLine(int fd, std::string& out, size_t cap = 65536) {
    out.clear();
    char buf[256];
    while (true) {
        ssize_t r = ::read(fd, buf, sizeof(buf));
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) {  // EOF before newline
            return false;
        }
        for (ssize_t i = 0; i < r; ++i) {
            if (buf[i] == '\n') {
                out.append(buf, i);
                // Discard rest of read (should be empty for line protocol)
                return true;
            }
        }
        out.append(buf, static_cast<size_t>(r));
        if (out.size() > cap) return false;  // safety
    }
}

// Send a SOLVE request and wait for DONE. Returns true on success and sets rc.
// On any IPC failure, tears down server state (next call respawns).
bool CallPersistentSolver(PersistentSolverState& s,
                          const std::string& json_dir,
                          const std::string& output_path,
                          const std::vector<std::string>& extra_args,
                          int& rc_out) {
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!EnsurePersistentSolver_NoLock(s)) return false;

    std::ostringstream cmd;
    cmd << "SOLVE\t" << json_dir << "\t" << output_path << "\t";
    for (size_t i = 0; i < extra_args.size(); ++i) {
        if (i) cmd << ' ';
        cmd << extra_args[i];
    }
    cmd << "\n";
    const std::string req = cmd.str();
    if (!WriteAll(s.fd, req.data(), req.size())) {
        TeardownPersistentSolver_NoLock(s, "write failed");
        return false;
    }
    std::string line;
    if (!ReadLine(s.fd, line)) {
        TeardownPersistentSolver_NoLock(s, "read EOF");
        return false;
    }
    if (line.rfind("DONE\t", 0) != 0) {
        TeardownPersistentSolver_NoLock(s, ("bad reply: " + line).c_str());
        return false;
    }
    try {
        rc_out = std::stoi(line.substr(5));
    } catch (...) {
        TeardownPersistentSolver_NoLock(s, "bad rc");
        return false;
    }
    return true;
}

// atexit: gracefully send QUIT and reap. Called once at process exit.
void ShutdownPersistentSolverAtExit() {
    auto& s = GetPersistentSolverState();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (s.fd >= 0) {
        const char* msg = "QUIT\n";
        WriteAll(s.fd, msg, std::strlen(msg));
    }
    TeardownPersistentSolver_NoLock(s, "atexit");
}

bool PersistentSolverEnabled() {
    const char* e = std::getenv("CPU_QR_USE_PERSISTENT_SOLVER");
    return (e && std::string(e) == "1");
}

Eigen::Matrix3f ExpSO3(const Eigen::Vector3f& omega) {
    const float theta = omega.norm();
    if (theta < 1e-8f) return Eigen::Matrix3f::Identity();
    Eigen::Matrix3f omega_hat;
    omega_hat << 0.0f, -omega(2), omega(1),
                 omega(2), 0.0f, -omega(0),
                 -omega(1), omega(0), 0.0f;
    return Eigen::Matrix3f::Identity()
        + (std::sin(theta) / theta) * omega_hat
        + ((1.0f - std::cos(theta)) / (theta * theta)) * omega_hat * omega_hat;
}

void UpsertPose(DumpPayload& payload, KeyFrame* kf) {
    if (!kf || kf->isBad()) return;
    const int pose_id = static_cast<int>(kf->mnId);
    // KeyFrame::GetVelocity() returns world-frame body velocity. When IMU is
    // not initialized yet it stays at zero — that's fine, deltaT will fall
    // back to "zero-velocity" assumption only for the pre-IMU phase.
    Eigen::Vector3f vw = kf->GetVelocity();
    // IMU bias: feed real bias into solver-side preintegration so deltaR /
    // deltaT / deltaV are bias-corrected (default GTSAM bias is zero which is
    // wrong once ORB has estimated a real bias).
    IMU::Bias b = kf->GetImuBias();
    Eigen::Vector3f ba(b.bax, b.bay, b.baz);
    Eigen::Vector3f bg(b.bwx, b.bwy, b.bwz);
    for (auto& p : payload.poses) {
        if (p.pose_id == pose_id) {
            p.Rcw = kf->GetRotation();
            p.tcw = kf->GetTranslation();
            p.vw  = vw;
            p.ba  = ba;
            p.bg  = bg;
            return;
        }
    }
    DumpPose p;
    p.pose_id = pose_id;
    p.Rcw = kf->GetRotation();
    p.tcw = kf->GetTranslation();
    p.vw  = vw;
    p.ba  = ba;
    p.bg  = bg;
    payload.poses.push_back(p);
}

void AddFixedPosePriors(DumpPayload& payload, const std::list<KeyFrame*>& fixed_kfs) {
    const char* env = std::getenv("CPU_QR_FIXED_SIGMA");
    const float sigma = env ? std::stof(env) : 1e-3f;
    for (KeyFrame* kf : fixed_kfs) {
        if (!kf || kf->isBad()) continue;
        DumpPrior p;
        p.pose_id = static_cast<int>(kf->mnId);
        p.Rp = kf->GetRotation();
        p.tp = kf->GetTranslation();
        p.sigma = Eigen::Matrix<float, 6, 1>::Constant(sigma);
        payload.priors.push_back(p);
    }
}

bool RunCpuQrSolver(const std::string& json_dir,
                    const SolverSwitch& sw,
                    const std::string& output_path,
                    const std::string& log_path) {
    // CPU QR solver invocation.
    // Default behavior (interleaved binary): minimal flags <json_dir> --output --huber --max-trails [--n-iter]
    // Legacy measure_cpu_qr binary: set CPU_QR_LEGACY_FLAGS=1 to also pass --online --tsqr --col-phase.
    // The first 3 args (binary, json_dir, --output) are positional/required.
    // We split "extra_args" so the persistent IPC path can rebuild argv on the
    // server side; the fork+exec path concatenates them all.
    std::vector<std::string> extra_args;
    extra_args.push_back("--huber=2.5");
    extra_args.push_back("--max-trails=" + std::to_string(sw.hw_max_trails));
    const char* legacy = std::getenv("CPU_QR_LEGACY_FLAGS");
    if (legacy && std::string(legacy) == "1") {
        // measure_cpu_qr (legacy landmark-first binary) needs these flags.
        extra_args.push_back("--online");
        const char* direct = std::getenv("CPU_QR_DIRECT");
        if (!direct || std::string(direct) != "1") {
            extra_args.push_back("--tsqr");
            extra_args.push_back("--col-phase");
        }
    }
    // Optional: pass --n-iter=N to solver if CPU_QR_N_ITER env var is set.
    // Default (unset) -> solver runs its own default (1 for interleaved binary).
    const char* n_iter = std::getenv("CPU_QR_N_ITER");
    if (n_iter && n_iter[0] != '\0') {
        extra_args.push_back(std::string("--n-iter=") + n_iter);
    }
    // Optional: arbitrary extra args appended verbatim (space-separated).
    // Used by persistent-solver mode to replace shell wrapper scripts.
    const char* extra_env = std::getenv("CPU_QR_EXTRA_ARGS");
    if (extra_env && extra_env[0] != '\0') {
        std::istringstream iss(extra_env);
        std::string tok;
        while (iss >> tok) extra_args.push_back(tok);
    }

    // ------------------------------------------------------------------
    // Path A: persistent solver IPC (preferred when enabled). Falls through
    // to fork+exec if the server is unreachable / dies mid-call.
    // ------------------------------------------------------------------
    if (PersistentSolverEnabled()) {
        auto& s = GetPersistentSolverState();
        int rc = -1;
        if (CallPersistentSolver(s, json_dir, output_path, extra_args, rc)) {
            if (rc == 0) return true;
            std::cerr << "[CPU_QR_PERSIST] solver returned rc=" << rc
                      << ", falling back to fork+exec\n";
        } else {
            std::cerr << "[CPU_QR_PERSIST] IPC failure, falling back to fork+exec\n";
        }
        // fall through to fork+exec
    }

    // ------------------------------------------------------------------
    // Path B: original fork+exec (always available as fallback).
    // ------------------------------------------------------------------
    std::vector<std::string> args;
    args.push_back(CpuQrSolverPath());
    args.push_back(json_dir);
    args.push_back("--output=" + output_path);
    for (const auto& a : extra_args) args.push_back(a);

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        int fd = open(log_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) fd = open("/dev/null", O_WRONLY);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }
        execvp(argv[0], argv.data());
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool ParseCpuQrOutput(const std::string& output_path,
                      std::map<int, Eigen::Matrix<float, 6, 1>>& pose_deltas,
                      std::map<int, Eigen::Vector3f>& landmark_deltas,
                      std::map<int, Eigen::Vector3f>& velocity_deltas,
                      std::map<int, Eigen::Vector3f>& gyro_bias_deltas,
                      std::map<int, Eigen::Vector3f>& acc_bias_deltas) {
    std::ifstream in(output_path);
    if (!in.is_open()) return false;
    std::string line;
    // Track highest-K per id; output format is POSE_<K>ITER / LANDMARK_<K>ITER.
    // Choose the largest K seen (= cumulative delta after all iters).
    std::map<int, int> pose_best_k;
    std::map<int, int> lm_best_k;
    std::map<int, int> vel_best_k;
    std::map<int, int> bg_best_k;
    std::map<int, int> ba_best_k;
    auto extract_k = [](const std::string& tag, const std::string& prefix) -> int {
        // tag like "POSE_3ITER" or "POSE_10ITER"; prefix like "POSE_"
        if (tag.rfind(prefix, 0) != 0) return -1;
        const std::string suffix = tag.substr(prefix.size());
        // suffix like "3ITER" or "10ITER"
        const std::size_t iter_pos = suffix.find("ITER");
        if (iter_pos == std::string::npos) return -1;
        try { return std::stoi(suffix.substr(0, iter_pos)); }
        catch (...) { return -1; }
    };
    while (std::getline(in, line)) {
        if (line.rfind("POSE_", 0) == 0 && line.find("ITER ") != std::string::npos) {
            const std::size_t hash = line.find('#');
            if (hash == std::string::npos) continue;
            std::istringstream head(line.substr(0, hash));
            std::istringstream vals(line.substr(hash + 1));
            std::string tag;
            int id = -1;
            head >> tag >> id;
            const int k = extract_k(tag, "POSE_");
            if (k <= 0) continue;
            auto bk = pose_best_k.find(id);
            if (bk != pose_best_k.end() && bk->second >= k) continue;  // already have higher-K
            Eigen::Matrix<float, 6, 1> delta;
            for (int i = 0; i < 6; ++i) {
                if (!(vals >> delta(i))) return false;
            }
            pose_deltas[id] = delta;
            pose_best_k[id] = k;
        } else if (line.rfind("LANDMARK_", 0) == 0 && line.find("ITER ") != std::string::npos) {
            const std::size_t hash = line.find('#');
            if (hash == std::string::npos) continue;
            std::istringstream head(line.substr(0, hash));
            std::istringstream vals(line.substr(hash + 1));
            std::string tag;
            int id = -1;
            head >> tag >> id;
            const int k = extract_k(tag, "LANDMARK_");
            if (k <= 0) continue;
            auto bk = lm_best_k.find(id);
            if (bk != lm_best_k.end() && bk->second >= k) continue;
            Eigen::Vector3f delta;
            for (int i = 0; i < 3; ++i) {
                if (!(vals >> delta(i))) return false;
            }
            landmark_deltas[id] = delta;
            lm_best_k[id] = k;
        } else if ((line.rfind("VEL_", 0) == 0 || line.rfind("BG_", 0) == 0 || line.rfind("BA_", 0) == 0) &&
                   line.find("ITER ") != std::string::npos) {
            const std::size_t hash = line.find('#');
            if (hash == std::string::npos) continue;
            std::istringstream head(line.substr(0, hash));
            std::istringstream vals(line.substr(hash + 1));
            std::string tag;
            int id = -1;
            head >> tag >> id;
            const bool is_vel = tag.rfind("VEL_", 0) == 0;
            const bool is_bg = tag.rfind("BG_", 0) == 0;
            const int k = extract_k(tag, is_vel ? "VEL_" : (is_bg ? "BG_" : "BA_"));
            if (k <= 0) continue;
            std::map<int, int>& best_k = is_vel ? vel_best_k : (is_bg ? bg_best_k : ba_best_k);
            auto bk = best_k.find(id);
            if (bk != best_k.end() && bk->second >= k) continue;
            Eigen::Vector3f delta;
            for (int i = 0; i < 3; ++i) {
                if (!(vals >> delta(i))) return false;
            }
            if (is_vel) velocity_deltas[id] = delta;
            else if (is_bg) gyro_bias_deltas[id] = delta;
            else acc_bias_deltas[id] = delta;
            best_k[id] = k;
        }
    }
    return !pose_deltas.empty() || !landmark_deltas.empty() || !velocity_deltas.empty() ||
           !gyro_bias_deltas.empty() || !acc_bias_deltas.empty();
}

// Pose-side safety: any pose delta NaN or > max → reject whole call.
// (Pose updates are global; bad pose can corrupt entire local map.)
bool PoseDeltasLookSafe(const std::map<int, Eigen::Matrix<float, 6, 1>>& pose_deltas) {
    const char* env = std::getenv("CPU_QR_MAX_DELTA");
    const float max_pose_delta = env ? std::stof(env) : 5.0f;
    for (const auto& kv : pose_deltas) {
        if (!kv.second.allFinite()) {
            std::cerr << "[CPU_QR_SAFE] pose " << kv.first << " not finite" << std::endl;
            return false;
        }
        float local_dt_norm = kv.second.head<3>().norm();
        if (local_dt_norm > max_pose_delta) {
            std::cerr << "[CPU_QR_SAFE] pose " << kv.first << " local_dt_norm=" << local_dt_norm << " > " << max_pose_delta << std::endl;
            return false;
        }
        float r_norm = kv.second.tail<3>().norm();
        if (r_norm > max_pose_delta) {
            std::cerr << "[CPU_QR_SAFE] pose " << kv.first << " r_norm=" << r_norm << " > " << max_pose_delta << std::endl;
            return false;
        }
    }
    return true;
}

bool PoseDeltasCenterMotionSafe(
    const std::list<KeyFrame*>& local_kfs,
    const std::map<int, Eigen::Matrix<float, 6, 1>>& pose_deltas) {
    const char* center_env = std::getenv("CPU_QR_ORB_MAX_CENTER_DELTA");
    const char* rot_env = std::getenv("CPU_QR_ORB_MAX_ROT_DELTA");
    if ((!center_env || center_env[0] == '\0') && (!rot_env || rot_env[0] == '\0')) return true;
    const float max_center_delta = (center_env && center_env[0] != '\0') ? std::stof(center_env) : std::numeric_limits<float>::infinity();
    const float max_rot_delta = (rot_env && rot_env[0] != '\0') ? std::stof(rot_env) : std::numeric_limits<float>::infinity();
    for (KeyFrame* kf : local_kfs) {
        if (!kf || kf->isBad()) continue;
        auto it = pose_deltas.find(static_cast<int>(kf->mnId));
        if (it == pose_deltas.end()) continue;
        Sophus::SE3f Tcw = kf->GetPose();
        Eigen::Matrix3f R0 = Tcw.rotationMatrix();
        Eigen::Vector3f t0 = Tcw.translation();
        Eigen::Vector3f dt = it->second.head<3>();
        Eigen::Vector3f omega = it->second.tail<3>();
        Eigen::Matrix3f R1 = R0 * ExpSO3(omega);
        Eigen::Vector3f t1 = t0 + R0 * dt;
        Eigen::Vector3f C0 = -R0.transpose() * t0;
        Eigen::Vector3f C1 = -R1.transpose() * t1;
        float center_norm = (C1 - C0).norm();
        float rot_norm = omega.norm();
        if (center_norm > max_center_delta || rot_norm > max_rot_delta) {
            std::cerr << "[CPU_QR_SAFE] pose " << kf->mnId
                      << " center_norm=" << center_norm << " max_center=" << max_center_delta
                      << " rot_norm=" << rot_norm << " max_rot=" << max_rot_delta << std::endl;
            return false;
        }
    }
    return true;
}

// Per-landmark safety: bad LM (NaN, huge norm) is filtered out individually.
// LMs are independent — outlier LMs shouldn't reject the whole call.
bool LandmarkDeltaSafe(const Eigen::Vector3f& delta) {
    if (!delta.allFinite()) return false;
    const char* lm_env = std::getenv("CPU_QR_LM_APPLY_MAX_DELTA");
    if (lm_env && lm_env[0] != '\0') {
        return delta.norm() <= std::stof(lm_env);
    }
    const char* env = std::getenv("CPU_QR_MAX_DELTA");
    const float max_pose_delta = env ? std::stof(env) : 5.0f;
    if (delta.norm() > 1000.0f * max_pose_delta) return false;
    return true;
}

float MedianValue(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

float ObservationReprojectionError(const DumpObservation& obs, const Eigen::Vector3f& Pw) {
    const Eigen::Vector3f pc = obs.Rcw * Pw + obs.tcw;
    if (!pc.allFinite() || pc.z() <= 1e-6f) return std::numeric_limits<float>::infinity();
    const float inv_z = 1.0f / pc.z();
    const Eigen::Vector2f uv(obs.fx * pc.x() * inv_z + obs.cx,
                             obs.fy * pc.y() * inv_z + obs.cy);
    const float sigma = (std::isfinite(obs.sigma_pixel) && obs.sigma_pixel > 1e-6f) ? obs.sigma_pixel : 1.0f;
    return (uv - obs.pixel).norm() / sigma;
}

bool LandmarkReprojectionWorse(const std::vector<const DumpObservation*>& observations,
                               const Eigen::Vector3f& delta,
                               float min_delta,
                               int max_obs) {
    if (delta.norm() <= min_delta || observations.empty()) return false;
    if (max_obs > 0 && observations.size() > static_cast<std::size_t>(max_obs)) return false;
    std::vector<float> before;
    std::vector<float> after;
    before.reserve(observations.size());
    after.reserve(observations.size());
    for (const DumpObservation* obs : observations) {
        before.push_back(ObservationReprojectionError(*obs, obs->landmark_w));
        after.push_back(ObservationReprojectionError(*obs, obs->landmark_w + delta));
    }
    return MedianValue(after) > MedianValue(before);
}

bool CpuQrVerboseCalls() {
    const char* env = std::getenv("CPU_QR_VERBOSE_CALLS");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool CpuQrEnvFlag(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

float CpuQrEnvFloat(const char* name, float default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') return default_value;
    return std::stof(env);
}

float MaxVecNorm(const std::map<int, Eigen::Vector3f>& deltas) {
    float max_norm = 0.0f;
    for (const auto& kv : deltas) max_norm = std::max(max_norm, kv.second.norm());
    return max_norm;
}

// Legacy combined check (kept for compatibility / diagnostic).
bool DeltaLooksSafe(const std::map<int, Eigen::Matrix<float, 6, 1>>& pose_deltas,
                    const std::map<int, Eigen::Vector3f>& landmark_deltas) {
    if (!PoseDeltasLookSafe(pose_deltas)) return false;
    for (const auto& kv : landmark_deltas) {
        if (!LandmarkDeltaSafe(kv.second)) {
            std::cerr << "[CPU_QR_SAFE] lm " << kv.first << " norm=" << kv.second.norm() << " unsafe" << std::endl;
            return false;
        }
    }
    return true;
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

    // Register persistent-solver shutdown hook the first time we enter.
    // No-op if the persistent path is never used.
    static std::once_flag s_atexit_once;
    std::call_once(s_atexit_once, []() {
        std::atexit(&ShutdownPersistentSolverAtExit);
    });

    auto cpu_qr_zero_update = [&]() -> LocalBAResult {
        result.success = sw.use_cpu_qr_solver;
        return result;
    };

    // ASIC stimulus dump for RTL replay: fires before gate so EVERY LBA call
    // (pre-VIBA2 native, post-VIBA2 CPU_QR, fallback) writes input JSON to
    // lba_kf_<id>/. Cost: FillExporter time on each call.
    auto dump_input_json = [&]() {
        if (!sw.dump_json) return;
        std::string dir = !json_dir.empty() ? json_dir : sw.json_out_dir;
        if (dir.empty()) dir = "/tmp/orbslam3_hw_dump";
        VariableEliminationDriver dump_driver;
        ORBSLAM3Exporter dump_exporter(dump_driver);
        FillExporter(input, dump_driver, dump_exporter);
        AddPriors(input, dump_exporter);
        DumpPayload payload = dump_exporter.snapshotPending(false);
        for (KeyFrame* kf : input.local_kfs) UpsertPose(payload, kf);
        for (KeyFrame* kf : input.fixed_kfs) UpsertPose(payload, kf);
        AddFixedPosePriors(payload, input.fixed_kfs);
        AsyncJsonWriter::Instance().Enqueue(std::move(payload), dir);
        g_has_pending_json = false;
    };
    dump_input_json();

    // CPU QR solver gate: run once IMU is initialized and post-VIBA2.
    if (sw.use_cpu_qr_solver) {
        bool inertial = input.map->IsInertial();
        bool imu_init = input.map->isImuInitialized();
        const char* allow_pre_ba2_env = std::getenv("CPU_QR_ALLOW_PRE_BA2");
        const bool allow_pre_ba2 = allow_pre_ba2_env && std::strcmp(allow_pre_ba2_env, "0") != 0;
        if (inertial && imu_init && !input.map->GetIniertialBA2() && !allow_pre_ba2) {
            static std::atomic<int> s_init_fallback_count{0};
            int n = s_init_fallback_count.fetch_add(1);
            if (n < 20 || CpuQrVerboseCalls()) {
                std::cerr << "[CPU_QR_GATE_NATIVE_INIT] #" << n
                          << " current_kf=" << input.current_kf->mnId
                          << " local_kfs=" << input.local_kfs.size()
                          << " fixed_kfs=" << input.fixed_kfs.size()
                          << " mps=" << input.local_mps.size() << std::endl;
            }
            return result;
        }
        if (inertial && imu_init) {
            static std::atomic<int> s_pass_count{0};
            int n = s_pass_count.fetch_add(1);
            if (n < 50 || CpuQrVerboseCalls()) {
                std::cerr << "[CPU_QR_GATE_PASS] #" << n
                          << " current_kf=" << input.current_kf->mnId
                          << " local_kfs=" << input.local_kfs.size()
                          << " fixed_kfs=" << input.fixed_kfs.size()
                          << " mps=" << input.local_mps.size()
                          << " pose_only=" << input.pose_only << std::endl;
            }
        }
        if (!inertial || !imu_init) {
            return cpu_qr_zero_update();
        }
    }

    VariableEliminationDriver driver;
    ORBSLAM3Exporter exporter(driver);

    // 填充因子
    FillExporter(input, driver, exporter);
    AddPriors(input, exporter);

    if (sw.use_cpu_qr_solver) {
        namespace fs = std::filesystem;
        static std::atomic<int> s_cpu_qr_seq{0};
        int seq_num = s_cpu_qr_seq.fetch_add(1);

        DumpPayload payload = exporter.snapshotPending(false);
        for (KeyFrame* kf : input.local_kfs) UpsertPose(payload, kf);
        for (KeyFrame* kf : input.fixed_kfs) UpsertPose(payload, kf);
        AddFixedPosePriors(payload, input.fixed_kfs);
        if (payload.poses.empty() || payload.observations.empty()) {
            return cpu_qr_zero_update();
        }

        // Per-call unique scratch dir to avoid race when multiple threads
        // (Tracking/LocalMapping/LoopClosing) call RunLocalBA concurrently.
        // Otherwise JSON files get overwritten mid-fork and solver reads
        // corrupted input → produces large/NaN deltas → safety check rejects.
        std::ostringstream tid_oss; tid_oss << std::this_thread::get_id();
        std::string scratch_dir = CpuQrScratchRoot() + "/pid_" +
                                  std::to_string(static_cast<long long>(getpid())) +
                                  "/tid_" + tid_oss.str() +
                                  "/call_" + std::to_string(seq_num);
        std::string output_path = scratch_dir + "/delta_x.txt";
        std::string delta_archive = scratch_dir + "/delta_x_" + std::to_string(seq_num) + ".txt";
        std::string log_path = scratch_dir + "/measure_cpu_qr.log";
        const bool verbose_calls = CpuQrVerboseCalls();
        if (verbose_calls) {
            std::cerr << "[CPU_QR_CALL_BEGIN] seq=" << seq_num
                      << " current_kf=" << input.current_kf->mnId
                      << " local_kfs=" << input.local_kfs.size()
                      << " fixed_kfs=" << input.fixed_kfs.size()
                      << " local_mps=" << input.local_mps.size()
                      << " scratch=" << scratch_dir << std::endl;
        }
        try {
            fs::create_directories(scratch_dir);
            DumpAllJson(payload, scratch_dir);
        } catch (const std::exception& e) {
            std::cerr << "[CPU_QR] failed to write JSON: " << e.what() << std::endl;
            return cpu_qr_zero_update();
        }

        if (!RunCpuQrSolver(scratch_dir, sw, output_path, log_path)) {
            std::cerr << "[CPU_QR] solver failed, log: " << log_path << std::endl;
            return cpu_qr_zero_update();
        }

        // 保存本次 delta 副本（防止被下次覆盖）
        try { fs::copy_file(output_path, delta_archive, fs::copy_options::overwrite_existing); }
        catch (...) {}

        std::map<int, Eigen::Matrix<float, 6, 1>> pose_deltas;
        std::map<int, Eigen::Vector3f> landmark_deltas;
        std::map<int, Eigen::Vector3f> velocity_deltas;
        std::map<int, Eigen::Vector3f> gyro_bias_deltas;
        std::map<int, Eigen::Vector3f> acc_bias_deltas;
        if (!ParseCpuQrOutput(output_path, pose_deltas, landmark_deltas,
                              velocity_deltas, gyro_bias_deltas, acc_bias_deltas)) {
            std::cerr << "[CPU_QR] failed to parse output: " << output_path << std::endl;
            return cpu_qr_zero_update();
        }

        if (verbose_calls) {
            float pose_dt_max = 0.0f;
            float pose_r_max = 0.0f;
            std::vector<float> lm_norms;
            lm_norms.reserve(landmark_deltas.size());
            for (const auto& kv : pose_deltas) {
                pose_dt_max = std::max(pose_dt_max, kv.second.head<3>().norm());
                pose_r_max = std::max(pose_r_max, kv.second.tail<3>().norm());
            }
            for (const auto& kv : landmark_deltas) {
                lm_norms.push_back(kv.second.norm());
            }
            float lm_max = 0.0f;
            float lm_median = 0.0f;
            int lm_ge_0p99 = 0;
            if (!lm_norms.empty()) {
                lm_max = *std::max_element(lm_norms.begin(), lm_norms.end());
                std::vector<float> tmp = lm_norms;
                std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
                lm_median = tmp[tmp.size() / 2];
                for (float n : lm_norms) {
                    if (n >= 0.99f) ++lm_ge_0p99;
                }
            }
            std::cerr << "[CPU_QR_CALL_PARSED] seq=" << seq_num
                      << " current_kf=" << input.current_kf->mnId
                      << " poses=" << pose_deltas.size()
                      << " landmarks=" << landmark_deltas.size()
                      << " velocities=" << velocity_deltas.size()
                      << " gyro_biases=" << gyro_bias_deltas.size()
                      << " acc_biases=" << acc_bias_deltas.size()
                      << " pose_dt_max=" << pose_dt_max
                      << " pose_r_max=" << pose_r_max
                      << " lm_max=" << lm_max
                      << " lm_median=" << lm_median
                      << " lm_ge_0p99=" << lm_ge_0p99
                      << " output=" << output_path << std::endl;
        }

        const bool diag_delta_stats = CpuQrEnvFlag("CPU_QR_DIAG_DELTA_STATS") || verbose_calls;
        const bool skip_velocity_apply = CpuQrEnvFlag("CPU_QR_SKIP_VELOCITY_APPLY");
        const bool skip_bias_apply = CpuQrEnvFlag("CPU_QR_SKIP_BIAS_APPLY");
        const float velocity_apply_gain = CpuQrEnvFloat("CPU_QR_VELOCITY_APPLY_GAIN", 1.0f);
        const float bias_apply_gain = CpuQrEnvFloat("CPU_QR_BIAS_APPLY_GAIN", 1.0f);
        if (diag_delta_stats) {
            float pose_dt_max = 0.0f;
            float pose_r_max = 0.0f;
            for (const auto& kv : pose_deltas) {
                pose_dt_max = std::max(pose_dt_max, kv.second.head<3>().norm());
                pose_r_max = std::max(pose_r_max, kv.second.tail<3>().norm());
            }
            std::cerr << "[CPU_QR_DELTA_STATS] seq=" << seq_num
                      << " pose_dt_max=" << pose_dt_max
                      << " pose_r_max=" << pose_r_max
                      << " vel_max=" << MaxVecNorm(velocity_deltas)
                      << " bg_max=" << MaxVecNorm(gyro_bias_deltas)
                      << " ba_max=" << MaxVecNorm(acc_bias_deltas)
                      << " velocity_gain=" << velocity_apply_gain
                      << " bias_gain=" << bias_apply_gain
                      << " skip_velocity=" << skip_velocity_apply
                      << " skip_bias=" << skip_bias_apply << std::endl;
        }

        if (!PoseDeltasLookSafe(pose_deltas) || !PoseDeltasCenterMotionSafe(input.local_kfs, pose_deltas)) {
            std::cerr << "[CPU_QR] unsafe pose delta (seq=" << seq_num << "), using zero update: " << output_path << std::endl;
            return cpu_qr_zero_update();
        }

        const char* min_lm_apply_env = std::getenv("CPU_QR_MIN_LM_APPLY_RATIO");
        const float min_lm_apply_ratio = min_lm_apply_env ? std::stof(min_lm_apply_env) : 0.0f;
        bool skip_all_landmarks = false;
        int lm_safe_count = 0;
        for (const auto& kv : landmark_deltas) {
            if (LandmarkDeltaSafe(kv.second)) ++lm_safe_count;
        }
        if (min_lm_apply_ratio > 0.0f && !landmark_deltas.empty()) {
            const float safe_ratio = static_cast<float>(lm_safe_count) / static_cast<float>(landmark_deltas.size());
            if (safe_ratio < min_lm_apply_ratio) {
                skip_all_landmarks = true;
                std::cerr << "[CPU_QR] seq=" << seq_num << " LM safe ratio " << safe_ratio
                          << " < " << min_lm_apply_ratio << ", skipping landmark updates: " << output_path << std::endl;
            }
        }

        std::map<int, std::vector<const DumpObservation*>> observations_by_landmark;
        const char* lm_reproj_gate_env = std::getenv("CPU_QR_LM_REPROJ_GATE");
        const bool lm_reproj_gate = lm_reproj_gate_env && lm_reproj_gate_env[0] != '\0' && std::strcmp(lm_reproj_gate_env, "0") != 0;
        const char* lm_gate_min_delta_env = std::getenv("CPU_QR_LM_GATE_MIN_DELTA");
        const char* lm_gate_max_obs_env = std::getenv("CPU_QR_LM_GATE_MAX_OBS");
        const float lm_gate_min_delta = lm_gate_min_delta_env ? std::stof(lm_gate_min_delta_env) : 3.0f;
        const int lm_gate_max_obs = lm_gate_max_obs_env ? std::stoi(lm_gate_max_obs_env) : 6;
        if (lm_reproj_gate) {
            for (const DumpObservation& obs : payload.observations) {
                observations_by_landmark[obs.landmark_id].push_back(&obs);
            }
        }

        const bool skip_current_pose_apply = std::getenv("CPU_QR_SKIP_CURRENT_POSE_APPLY") != nullptr;
        const bool skip_max_local_pose_apply = std::getenv("CPU_QR_SKIP_MAX_LOCAL_POSE_APPLY") != nullptr;
        const char* damp_dt_env = std::getenv("CPU_QR_POSE_DAMP_DT");
        const char* damp_rot_env = std::getenv("CPU_QR_POSE_DAMP_ROT");
        const float pose_damp_dt = damp_dt_env ? std::strtof(damp_dt_env, nullptr) : 0.0f;
        const float pose_damp_rot = damp_rot_env ? std::strtof(damp_rot_env, nullptr) : 0.0f;
        int max_local_pose_id = -1;
        if (skip_max_local_pose_apply) {
            for (KeyFrame* kf : input.local_kfs) {
                if (!kf || kf->isBad()) continue;
                max_local_pose_id = std::max(max_local_pose_id, static_cast<int>(kf->mnId));
            }
        }
        bool logged_pose_skip = false;

        for (KeyFrame* kf : input.local_kfs) {
            if (!kf || kf->isBad()) continue;
            const int kid = static_cast<int>(kf->mnId);
            auto it = pose_deltas.find(kid);
            if (it != pose_deltas.end()) {
                const bool skip_pose_apply = (skip_current_pose_apply && kf == input.current_kf) ||
                                             (skip_max_local_pose_apply && kid == max_local_pose_id);
                if (skip_pose_apply) {
                    if (!logged_pose_skip) {
                        std::cerr << "[CPU_QR_POSE_SKIP] seq=" << seq_num
                                  << " current_kf=" << input.current_kf->mnId
                                  << " skipped_kf=" << kid
                                  << " skip_current=" << skip_current_pose_apply
                                  << " skip_max_local=" << skip_max_local_pose_apply << std::endl;
                        logged_pose_skip = true;
                    }
                } else {
                    Eigen::Vector3f dt = it->second.head<3>();
                    Eigen::Vector3f omega = it->second.tail<3>();
                    const float dt_norm = dt.norm();
                    const float omega_norm = omega.norm();
                    bool damped_pose = false;
                    if (pose_damp_dt > 0.0f && dt_norm > pose_damp_dt) {
                        dt *= pose_damp_dt / dt_norm;
                        damped_pose = true;
                    }
                    if (pose_damp_rot > 0.0f && omega_norm > pose_damp_rot) {
                        omega *= pose_damp_rot / omega_norm;
                        damped_pose = true;
                    }
                    if (damped_pose) {
                        std::cerr << "[CPU_QR_POSE_DAMP] seq=" << seq_num
                                  << " kf=" << kid
                                  << " dt_before=" << dt_norm
                                  << " rot_before=" << omega_norm
                                  << " dt_after=" << dt.norm()
                                  << " rot_after=" << omega.norm() << std::endl;
                    }
                    Sophus::SE3f Tcw = kf->GetPose();
                    Eigen::Matrix3f dR = ExpSO3(omega);
                    Eigen::Matrix3f R0 = Tcw.rotationMatrix();
                    Eigen::Matrix3f Rcw = R0 * dR;
                    // Re-orthogonalize after FP32 multiplication (Sophus requires strict orthogonality)
                    Eigen::JacobiSVD<Eigen::Matrix3f> svd(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Rcw = svd.matrixU() * svd.matrixV().transpose();
                    Eigen::Vector3f tcw = Tcw.translation() + R0 * dt;
                    result.pose_updates[kf] = Sophus::SE3f(Rcw, tcw);
                }
            }
            if (!skip_velocity_apply) {
                if (auto vit = velocity_deltas.find(kid); vit != velocity_deltas.end()) {
                    result.velocity_updates[kf] = kf->GetVelocity() + velocity_apply_gain * vit->second;
                }
            }
            if (!skip_bias_apply) {
                auto bg_it = gyro_bias_deltas.find(kid);
                auto ba_it = acc_bias_deltas.find(kid);
                if (bg_it != gyro_bias_deltas.end() || ba_it != acc_bias_deltas.end()) {
                    IMU::Bias b = kf->GetImuBias();
                    Eigen::Vector3f bg(b.bwx, b.bwy, b.bwz);
                    Eigen::Vector3f ba(b.bax, b.bay, b.baz);
                    if (bg_it != gyro_bias_deltas.end()) bg += bias_apply_gain * bg_it->second;
                    if (ba_it != acc_bias_deltas.end()) ba += bias_apply_gain * ba_it->second;
                    result.bias_updates[kf] = IMU::Bias(ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z());
                }
            }
        }
        int lm_skipped = 0;
        int lm_reproj_skipped = 0;
        if (skip_all_landmarks) {
            lm_skipped = static_cast<int>(landmark_deltas.size());
        } else {
            for (MapPoint* mp : input.local_mps) {
                if (!mp || mp->isBad()) continue;
                const int lm_id = static_cast<int>(mp->mnId);
                auto it = landmark_deltas.find(lm_id);
                if (it != landmark_deltas.end()) {
                    if (!LandmarkDeltaSafe(it->second)) {
                        ++lm_skipped;
                        continue;  // skip outlier LM, keep its current world pos
                    }
                    if (lm_reproj_gate) {
                        auto obs_it = observations_by_landmark.find(lm_id);
                        const std::vector<const DumpObservation*> no_observations;
                        const std::vector<const DumpObservation*>& obs = (obs_it != observations_by_landmark.end()) ? obs_it->second : no_observations;
                        if (LandmarkReprojectionWorse(obs, it->second, lm_gate_min_delta, lm_gate_max_obs)) {
                            ++lm_skipped;
                            ++lm_reproj_skipped;
                            continue;
                        }
                    }
                    result.landmark_updates[mp] = mp->GetWorldPos().cast<float>() + it->second;
                }
            }
        }
        if (lm_skipped > 0) {
            static std::atomic<int> s_lm_skip_log{0};
            if (s_lm_skip_log.fetch_add(1) < 20 || verbose_calls) {
                std::cerr << "[CPU_QR] seq=" << seq_num << " skipped " << lm_skipped
                          << " outlier LM updates (reproj=" << lm_reproj_skipped
                          << ", kept poses)" << std::endl;
            }
        }
        if (verbose_calls) {
            std::cerr << "[CPU_QR_CALL_APPLY] seq=" << seq_num
                      << " current_kf=" << input.current_kf->mnId
                      << " pose_updates=" << result.pose_updates.size()
                      << " velocity_updates=" << result.velocity_updates.size()
                      << " bias_updates=" << result.bias_updates.size()
                      << " landmark_updates=" << result.landmark_updates.size()
                      << " lm_skipped=" << lm_skipped
                      << " lm_reproj_skipped=" << lm_reproj_skipped << std::endl;
        }
        result.success = true;
        return result;
    }

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

    struct CandidateObservation {
        MapPoint* mp = nullptr;
        KeyFrame* kf = nullptr;
        int idx = -1;
        int local_track = 0;
        int local_hits = 0;
        int fixed_hits = 0;
        int score = 0;
    };

    std::vector<CandidateObservation> selected_obs;
    std::set<MapPoint*> selected_mps;
    if (!input.pose_only) {
        std::vector<CandidateObservation> candidates;
        for (MapPoint* mp : input.local_mps) {
            if (!mp || mp->isBad()) continue;
            const auto& obs = mp->GetObservations();
            std::vector<CandidateObservation> mp_obs;
            int local_hits = 0;
            int fixed_hits = 0;
            for (const auto& kv : obs) {
                KeyFrame* kf = kv.first;
                if (!kf || kf->isBad()) continue;
                const bool is_local = IsKeyFrameInSet(kf, local_set);
                const bool is_fixed = IsKeyFrameInSet(kf, fixed_set);
                if (!is_local && !is_fixed) continue;
                const int idx = std::get<0>(kv.second);
                if (idx < 0 || idx >= static_cast<int>(kf->mvKeysUn.size())) continue;
                if (is_local) ++local_hits;
                if (is_fixed) ++fixed_hits;
                CandidateObservation co;
                co.mp = mp;
                co.kf = kf;
                co.idx = idx;
                mp_obs.push_back(co);
            }
            if (local_hits == 0 || mp_obs.empty()) continue;
            const int track = static_cast<int>(mp_obs.size());
            const int obs_keep = std::min(track, ASIC_OBS_PER_LANDMARK_CAP);
            std::sort(mp_obs.begin(), mp_obs.end(),
                      [input](const CandidateObservation& a, const CandidateObservation& b) {
                          const bool ac = (a.kf == input.current_kf);
                          const bool bc = (b.kf == input.current_kf);
                          if (ac != bc) return ac;
                          return a.kf->mnId > b.kf->mnId;
                      });
            for (int i = 0; i < obs_keep; ++i) {
                CandidateObservation co = mp_obs[static_cast<std::size_t>(i)];
                co.local_track = track;
                co.local_hits = local_hits;
                co.fixed_hits = fixed_hits;
                co.score = 1000 * local_hits + 100 * fixed_hits + track;
                candidates.push_back(co);
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const CandidateObservation& a, const CandidateObservation& b) {
                      if (a.kf->mnId != b.kf->mnId) return a.kf->mnId > b.kf->mnId;
                      if (a.score != b.score) return a.score > b.score;
                      return a.mp->mnId < b.mp->mnId;
                  });

        std::map<KeyFrame*, int> kept_per_kf;
        std::map<MapPoint*, int> kept_per_mp;
        for (const CandidateObservation& co : candidates) {
            if (kept_per_kf[co.kf] >= ASIC_FEATURES_PER_KF_CAP) continue;
            if (kept_per_mp[co.mp] >= ASIC_OBS_PER_LANDMARK_CAP) continue;
            selected_obs.push_back(co);
            selected_mps.insert(co.mp);
            ++kept_per_kf[co.kf];
            ++kept_per_mp[co.mp];
        }

        if (selected_mps.size() > static_cast<std::size_t>(LANDMARK_CAP)) {
            std::vector<MapPoint*> ranked_mps(selected_mps.begin(), selected_mps.end());
            std::sort(ranked_mps.begin(), ranked_mps.end(),
                      [&kept_per_mp](MapPoint* a, MapPoint* b) {
                          const int ka = kept_per_mp[a];
                          const int kb = kept_per_mp[b];
                          if (ka != kb) return ka > kb;
                          return a->Observations() > b->Observations();
                      });
            std::set<MapPoint*> limited_mps;
            for (std::size_t i = 0; i < static_cast<std::size_t>(LANDMARK_CAP) && i < ranked_mps.size(); ++i) {
                limited_mps.insert(ranked_mps[i]);
            }
            selected_mps.swap(limited_mps);
            selected_obs.erase(std::remove_if(selected_obs.begin(), selected_obs.end(),
                                             [&selected_mps](const CandidateObservation& co) {
                                                 return selected_mps.find(co.mp) == selected_mps.end();
                                             }),
                               selected_obs.end());
        }
    }

    // 2) 路标初值
    if (!input.pose_only) {
        for (MapPoint* mp : selected_mps) {
            driver.setLandmark(mp->mnId, mp->GetWorldPos().cast<float>());
        }
    }

    // 3) 相机观测因子（位姿-路标因子），若是 pose-only 模式则跳过
    if (!input.pose_only) {
        for (const CandidateObservation& co : selected_obs) {
            KeyFrame* kf = co.kf;
            MapPoint* mp = co.mp;
            const cv::KeyPoint& kp = kf->mvKeysUn[co.idx];

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
            Eigen::Matrix<double,2,1> obs2d;
            obs2d << kp.pt.x, kp.pt.y;
            const float unc2 = kf->mpCamera->uncertainty2(obs2d);
            const float invSigma2 = kf->mvInvLevelSigma2[kp.octave] / unc2;
            oi.sigma_pixel = 1.0f / std::sqrt(invSigma2);

            exporter.addObservation(oi);
        }
    }

    // 4) 暂不添加 IMU 因子（需按实际 IMU 流水对接）
    // 4) IMU 因子（使用 KeyFrame 内部预积分）；仅当不是纯 pose-only 图
    if (input.inertial && !input.pose_only) {
        for (KeyFrame* kf : input.local_kfs) {
            if (!kf || !kf->mpImuPreintegrated || !kf->mPrevKF) continue;
            KeyFrame* kf_prev = kf->mPrevKF;
            if (kf_prev->GetMap() != kf->GetMap()) continue;

            // 构造原始 IMU 序列
            std::vector<ImuPreprocessorGTSAM::ImuMeasurement> seg;
            auto* pim = kf->mpImuPreintegrated;
            const auto& meas = pim->GetMeasurements();
            if (!meas.empty()) {
                for (const auto& z : meas) {
                    double dt = z.t;
                    if (dt <= 0) continue;
                    ImuPreprocessorGTSAM::ImuMeasurement m;
                    m.acc = z.a.cast<double>();
                    m.gyro = z.w.cast<double>();
                    m.dt = dt;
                    seg.push_back(m);
                }
            }
            if (seg.empty()) continue;

            double dt_total = pim->dT;

            ORBSLAM3Exporter::ImuEdgeInput edge;
            edge.pose_i_id = kf_prev->mnId;
            edge.pose_j_id = kf->mnId;
            edge.measurements = std::move(seg);
            edge.dt_total = dt_total;
            edge.has_preintegrated = true;
            edge.dR = pim->dR;
            edge.dV = pim->dV;
            edge.dP = pim->dP;
            edge.JRg = pim->JRg;
            edge.JVg = pim->JVg;
            edge.JVa = pim->JVa;
            edge.JPg = pim->JPg;
            edge.JPa = pim->JPa;
            IMU::Bias b0 = pim->GetOriginalBias();
            edge.ba0 = Eigen::Vector3f(b0.bax, b0.bay, b0.baz);
            edge.bg0 = Eigen::Vector3f(b0.bwx, b0.bwy, b0.bwz);
            edge.C = pim->C;

            // Use ORB-SLAM3 preintegration covariance for sigma (ground truth)
            // C(15x15): [theta(3), vel(3), pos(3), bg(3), ba(3)]
            // sigma order: [rotation(3), position(3)]
            const auto& C = pim->C;
            for (int k = 0; k < 3; ++k) {
                float var_r = C(k, k);
                float var_t = C(k + 6, k + 6);
                edge.sigma(k)     = std::sqrt(std::max(var_r, 1e-20f));
                edge.sigma(k + 3) = std::sqrt(std::max(var_t, 1e-20f));
            }

            // Use addImuEdge (not addImuConstraintDirect) to preserve raw measurements
            // for per-LBA JSON dump. sigma is passed through edge.sigma.
            exporter.addImuEdge(edge);
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


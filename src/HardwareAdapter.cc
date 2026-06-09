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
                    const std::string& log_path,
                    bool ba2_done_for_solver) {
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
    // Per-call BA2 state for solver-side trust-region gates. This must be an
    // argument instead of an environment variable so persistent solver mode sees
    // the current call's BA stage rather than the server process startup state.
    extra_args.push_back(std::string("--ba2-done=") + (ba2_done_for_solver ? "1" : "0"));
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

    // CPU QR solver: handle inertial LBA calls (imu_init=1).
    // Visual-only calls (imu_init=0) return zero-update → g2o fallback.
    if (sw.use_cpu_qr_solver) {
        const bool inertial = input.map->IsInertial();
        const bool imu_init = input.map->isImuInitialized();
        const bool ba2_done = input.map->GetIniertialBA2();
        // CPU_QR_REQUIRE_BA2: only let the ASIC own inertial LBA once the map is
        // mature (post InertialBA2 — scale & gravity refined). Engaging the
        // inertial ASIC right after BA1 (imu_init=1 but ba2=0) feeds it an
        // ill-conditioned, still-converging inertial map -> fp32+single-step
        // solver diverges -> map corruption -> reset storm. The historically
        // good run (full_inertial_n1: MH04=0.024m) only switched to ASIC at
        // fixed_kfs~35 (post-BA2); pre-BA2 LBA went to native g2o. This flag
        // reproduces that legitimate boundary: g2o handles bootstrap + IMU
        // init, ASIC owns the steady-state inertial Local BA.
        static const bool s_require_ba2 =
            std::getenv("CPU_QR_REQUIRE_BA2") != nullptr;
        // CPU_QR_PUREVIS_ASIC: true full replacement. Let the ASIC own the
        // pre-IMU-init pure-visual local BA too (imu_init=0) instead of gating it
        // to native/g2o. The solver drops vel/bg/ba columns for no-IMU KFs (6-DoF
        // block) so the visual-only window is well-posed; the init KF stays fixed
        // (Optimizer hook) to anchor the SE3 gauge. Default off -> identical gating,
        // so a rebuilt .so loaded by the safe PREINIT-to-g2o runs is unaffected.
        static const bool s_purevis_asic =
            std::getenv("CPU_QR_PUREVIS_ASIC") != nullptr;
        const bool gate_to_native = !inertial ||
                                    (!imu_init && !s_purevis_asic) ||
                                    (s_require_ba2 && !ba2_done);
        if (gate_to_native) {
            static std::atomic<int> s_native_count{0};
            int nn = s_native_count.fetch_add(1);
            if (nn < 20 || CpuQrVerboseCalls()) {
                std::cerr << "[CPU_QR_GATE_NATIVE] #" << nn
                          << " current_kf=" << input.current_kf->mnId
                          << " imu_init=" << (imu_init ? 1 : 0) << std::endl;
            }
            // Pre-IMU-init / visual-only BA is a 6-DoF problem with no inertial
            // states or factors -> the inertial ASIC cannot solve it and a
            // zero-update return disables bootstrap BA, which triggers tracking-
            // loss reset storms on harder sequences. With CPU_QR_PREINIT_TO_G2O
            // set, route these calls to native g2o (success=false -> the Optimizer
            // hook falls through) so the bootstrap map is properly optimized; the
            // ASIC still owns ALL inertial LBA (imu_init=1 -> GATE_PASS below).
            static const bool s_preinit_to_g2o =
                std::getenv("CPU_QR_PREINIT_TO_G2O") != nullptr;
            if (s_preinit_to_g2o) {
                if (nn < 20 || CpuQrVerboseCalls()) {
                    std::cerr << "[CPU_QR_GATE_NATIVE_G2O] #" << nn
                              << " -> native g2o (pre-init visual BA)" << std::endl;
                }
                result.success = false;  // fall through to native g2o in Optimizer
                return result;
            }
            return cpu_qr_zero_update();
        }
        // CPU_QR_G2O_REFRESH_EVERY=N: route every Nth post-BA2 inertial LBA to
        // native g2o (double precision) to correct the fp32 single-step gauge
        // drift that otherwise accumulates over hundreds of ASIC solves into
        // meters of ATE error (MH04 full=7.04m, MH01=4.16m). The ASIC still owns
        // (N-1)/N of the steady-state inertial Local BA; g2o is a low-frequency
        // gauge re-anchor (1/N). Large N keeps the ASIC share high while bounding
        // drift. The diverged-call count is logged as [CPU_QR_G2O_REFRESH] so the
        // ASIC/g2o split is auditable for the "legitimate replacement" claim.
        static const int s_refresh_every = []() {
            const char* e = std::getenv("CPU_QR_G2O_REFRESH_EVERY");
            return (e && e[0]) ? std::atoi(e) : 0;
        }();
        if (s_refresh_every > 0) {
            static std::atomic<int> s_refresh_count{0};
            int rc = s_refresh_count.fetch_add(1);
            if (rc % s_refresh_every == (s_refresh_every - 1)) {
                static std::atomic<int> s_refresh_diverts{0};
                int rd = s_refresh_diverts.fetch_add(1);
                if (rd < 200 || CpuQrVerboseCalls()) {
                    std::cerr << "[CPU_QR_G2O_REFRESH] #" << rd
                              << " (every " << s_refresh_every << ") -> native g2o"
                              << std::endl;
                }
                result.success = false;  // fall through to native g2o gauge refresh
                return result;
            }
        }
        // Strict full-replacement bootstrap quality gate. If the pure-visual map is
        // still too sparse, do not let an early ASIC LBA write into a poor basin; also
        // do not fall through to g2o. Return a successful zero-update and wait for the
        // front-end to add more KFs/points before the first ASIC write-back.
        const int min_preinit_mps = static_cast<int>(CpuQrEnvFloat("CPU_QR_MIN_PREINIT_MPS", 0.0f));
        if (!imu_init && min_preinit_mps > 0 &&
            static_cast<int>(input.local_mps.size()) < min_preinit_mps) {
            static std::atomic<int> s_preinit_sparse_skip{0};
            const int ss = s_preinit_sparse_skip.fetch_add(1);
            if (ss < 120 || CpuQrVerboseCalls()) {
                std::cerr << "[CPU_QR_PREINIT_SPARSE_SKIP] #" << ss
                          << " current_kf=" << input.current_kf->mnId
                          << " mps=" << input.local_mps.size()
                          << " min=" << min_preinit_mps
                          << " -> zero-update, no g2o" << std::endl;
            }
            return cpu_qr_zero_update();
        }
        static std::atomic<int> s_pass_count{0};
        int n = s_pass_count.fetch_add(1);
        if (n < 200 || CpuQrVerboseCalls()) {
            std::cerr << "[CPU_QR_GATE_PASS] #" << n
                      << " current_kf=" << input.current_kf->mnId
                      << " local_kfs=" << input.local_kfs.size()
                      << " fixed_kfs=" << input.fixed_kfs.size()
                      << " mps=" << input.local_mps.size()
                      << " inertial=" << (inertial ? 1 : 0)
                      << " imu_init=" << (imu_init ? 1 : 0)
                      << " ba2=" << (ba2_done ? 1 : 0)
                      << " pose_only=" << input.pose_only << std::endl;
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
        const bool ba2_done_for_solver = input.map && input.map->GetIniertialBA2();
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

        if (!RunCpuQrSolver(scratch_dir, sw, output_path, log_path, ba2_done_for_solver)) {
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
        // Anchor-drift guard (CPU_QR_ANCHOR_BIAS_FREEZE=1). Root cause of the cross-call
        // IMU-chi2 snowball in the 100% ASIC inertial LBA: the sliding window reselects its
        // FIXED anchor each call as vpOptimizableKFs.back()->mPrevKF, so the oldest free KFs
        // of THIS window become the FIXED gauge anchor of the NEXT window. While free they
        // received a full-gain fp32 single-step vel/bias writeback that drifts a little; that
        // drifted value is then frozen-in as the next window's hard anchor -> the absolute
        // velocity/accel-bias gauge walks off and IMU residual grows to 1e5-1e6 at entry.
        // g2o has no such issue because its anchor KF has V/VG/VA setFixed(true) and is never
        // re-solved. We emulate that by applying a STRONGLY reduced gain (default 0 = hard
        // freeze, like setFixed) to the vel/bias writeback of the oldest N_ANCHOR free KFs.
        // Pose is untouched (still 100% ASIC-solved) so the ATE source is unchanged; this
        // only damps the weakly-observable dynamic-state writeback that feeds the next anchor.
        const bool anchor_bias_freeze = CpuQrEnvFlag("CPU_QR_ANCHOR_BIAS_FREEZE");
        const float anchor_vel_gain = CpuQrEnvFloat("CPU_QR_ANCHOR_VEL_GAIN", 0.0f);
        const float anchor_bias_gain = CpuQrEnvFloat("CPU_QR_ANCHOR_BIAS_GAIN", 0.0f);
        const int anchor_n_kf = static_cast<int>(CpuQrEnvFloat("CPU_QR_ANCHOR_N_KF", 2.0f));
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

        // Per-pose-skip mode: instead of rejecting the WHOLE call when one pose
        // delta exceeds MAX_DELTA (which wastes the good poses), only NaN/inf in
        // any pose rejects the whole call; oversized individual poses are skipped
        // in the apply loop below (their current tracking value is kept).
        // When the ASIC produced an unusable solution for THIS call, a consistent
        // native g2o solve (success=false -> Optimizer hook falls through) keeps the
        // map healthier than a zero update, which leaves the window unoptimized and
        // accumulates drift until tracking is lost and the map collapses (empty /
        // tiny-fragment map -> huge ATE). With CPU_QR_G2O_SAFETY_FALLBACK set,
        // reject-to-g2o; otherwise keep the legacy zero-update behavior. The ASIC
        // still owns every call whose solution is safe -> primary inertial-LBA solver.
        const bool g2o_safety_fallback = CpuQrEnvFlag("CPU_QR_G2O_SAFETY_FALLBACK");
        auto safety_reject = [&](const std::string& why) -> LocalBAResult {
            std::cerr << "[CPU_QR_SAFETY_" << (g2o_safety_fallback ? "G2O" : "ZERO")
                      << "] seq=" << seq_num << " " << why << std::endl;
            if (g2o_safety_fallback) { result.success = false; return result; }
            return cpu_qr_zero_update();
        };

        const bool per_pose_skip = CpuQrEnvFlag("CPU_QR_PER_POSE_SKIP");
        const char* per_pose_max_env = std::getenv("CPU_QR_MAX_DELTA");
        const float per_pose_max_delta = per_pose_max_env ? std::stof(per_pose_max_env) : 5.0f;
        // Whole-call reject threshold (default off): if ANY pose translation delta
        // exceeds this, the ASIC solve diverged for this window -> route the whole
        // call to g2o rather than per-pose-skipping survivors into an inconsistent
        // state. Only meaningful together with CPU_QR_G2O_SAFETY_FALLBACK.
        const float call_reject_dt = CpuQrEnvFloat("CPU_QR_CALL_REJECT_DT", 0.0f);
        if (per_pose_skip) {
            bool pose_finite_ok = true;
            float pose_dt_max = 0.0f;
            for (const auto& kv : pose_deltas) {
                if (!kv.second.allFinite()) { pose_finite_ok = false; break; }
                pose_dt_max = std::max(pose_dt_max, kv.second.head<3>().norm());
            }
            if (!pose_finite_ok) {
                return safety_reject("non-finite pose delta");
            }
            if (g2o_safety_fallback && call_reject_dt > 0.0f && pose_dt_max > call_reject_dt) {
                return safety_reject("pose_dt_max " + std::to_string(pose_dt_max) +
                                     " > call_reject_dt " + std::to_string(call_reject_dt));
            }
        } else if (!PoseDeltasLookSafe(pose_deltas) || !PoseDeltasCenterMotionSafe(input.local_kfs, pose_deltas)) {
            return safety_reject("unsafe pose delta");
        }

        const char* min_lm_apply_env = std::getenv("CPU_QR_MIN_LM_APPLY_RATIO");
        const float min_lm_apply_ratio = min_lm_apply_env ? std::stof(min_lm_apply_env) : 0.0f;
        bool skip_all_landmarks = false;
        // During the pure-visual bootstrap, landmark write-back can perturb the visual map
        // before IMU scale/gravity/bias have even been initialized. Keep the ASIC solve, but
        // optionally make only this very early stage pose-only. Once IMU initialization starts,
        // restore landmark write-back so visual BA can keep driving BA1/BA2 into a good basin.
        const bool imu_init_for_lm_guard = input.map && input.map->isImuInitialized();
        const bool ba2_done_for_lm_guard = input.map && input.map->GetIniertialBA2();
        const bool guard_until_ba2 = CpuQrEnvFlag("CPU_QR_PREINIT_SKIP_LANDMARKS_UNTIL_BA2");
        const bool lm_guard_active = CpuQrEnvFlag("CPU_QR_PREINIT_SKIP_LANDMARKS") &&
                                     (guard_until_ba2 ? !ba2_done_for_lm_guard : !imu_init_for_lm_guard);
        if (lm_guard_active) {
            skip_all_landmarks = true;
            static std::atomic<int> s_preinit_lm_guard_log{0};
            const int nn = s_preinit_lm_guard_log.fetch_add(1);
            if (nn < 80) {
                std::cerr << "[CPU_QR_PREINIT_LM_GUARD] seq=" << seq_num
                          << " imu_init=" << (imu_init_for_lm_guard ? 1 : 0)
                          << " ba2=" << (ba2_done_for_lm_guard ? 1 : 0)
                          << " -> skip landmark writeback" << std::endl;
            }
        }
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
        const bool ba2_done_for_trust_region = input.map && input.map->GetIniertialBA2();
        const bool trust_region_pre_ba2_only = CpuQrEnvFlag("CPU_QR_TRUST_REGION_PRE_BA2_ONLY");
        const bool trust_region_active = !trust_region_pre_ba2_only || !ba2_done_for_trust_region;
        const float max_vel_total = trust_region_active ? CpuQrEnvFloat("CPU_QR_MAX_VEL_TOTAL", 0.0f) : 0.0f;
        const float max_bg_total = trust_region_active ? CpuQrEnvFloat("CPU_QR_MAX_BG_TOTAL", 0.0f) : 0.0f;
        const float max_ba_total = trust_region_active ? CpuQrEnvFloat("CPU_QR_MAX_BA_TOTAL", 0.0f) : 0.0f;
        auto clamp_dynamic_delta = [&](Eigen::Vector3f d, float cap, const char* kind, int kid) {
            if (cap > 0.0f) {
                const float before = d.norm();
                if (before > cap) {
                    d *= cap / before;
                    std::cerr << "[CPU_QR_DYN_CLAMP] seq=" << seq_num
                              << " kf=" << kid
                              << " kind=" << kind
                              << " before=" << before
                              << " after=" << d.norm()
                              << " cap=" << cap << std::endl;
                }
            }
            return d;
        };
        const char* damp_dt_env = std::getenv("CPU_QR_POSE_DAMP_DT");
        const char* damp_rot_env = std::getenv("CPU_QR_POSE_DAMP_ROT");
        const float pose_damp_dt = trust_region_active && damp_dt_env ? std::strtof(damp_dt_env, nullptr) : 0.0f;
        const float pose_damp_rot = trust_region_active && damp_rot_env ? std::strtof(damp_rot_env, nullptr) : 0.0f;
        // Trust-region style gain (<1) on applied pose delta to damp FP32 overshoot.
        const float pose_apply_gain = CpuQrEnvFloat("CPU_QR_POSE_APPLY_GAIN", 1.0f);
        static std::atomic<int> s_pp_skip_log{0};
        int max_local_pose_id = -1;
        if (skip_max_local_pose_apply) {
            for (KeyFrame* kf : input.local_kfs) {
                if (!kf || kf->isBad()) continue;
                max_local_pose_id = std::max(max_local_pose_id, static_cast<int>(kf->mnId));
            }
        }
        bool logged_pose_skip = false;

        // Anchor candidates = the oldest (smallest mnId) free KFs in this window; they will
        // become the FIXED anchor of a later sliding window. Collect their ids so the vel/bias
        // writeback below can freeze/damp them. Pose writeback is never affected.
        std::set<int> anchor_kf_ids;
        if (anchor_bias_freeze && anchor_n_kf > 0) {
            std::vector<int> ids;
            ids.reserve(input.local_kfs.size());
            for (KeyFrame* kf : input.local_kfs) {
                if (!kf || kf->isBad()) continue;
                ids.push_back(static_cast<int>(kf->mnId));
            }
            std::sort(ids.begin(), ids.end());
            for (int i = 0; i < anchor_n_kf && i < static_cast<int>(ids.size()); ++i)
                anchor_kf_ids.insert(ids[i]);
        }

        // Call-level circuit breaker: if THIS LBA call's deltas are physically implausible
        // (a tracking-failure-driven blow-up, e.g. a 0.38m single-call pose jump or a 0.19
        // accel-bias jump), reject the whole call's write-back and keep the front-end tracking
        // state. Native ORB-SLAM3 never folds a corrupt optimization into the map; its own
        // IMU-coast / relocalize then handles recovery instead of snowballing.
        const bool cb_enable = CpuQrEnvFlag("CPU_QR_CB_ENABLE");
        // The circuit breaker is a last-resort post-BA2 guard.  Pre-BA2 IMU scale/gravity/bias
        // are still settling; strict-good V1_01 accepts small ba steps around 0.067 before BA2.
        // Rejecting those early calls starves the inertial init chain and delays BA2 into a bad
        // basin.  Default: disable CB until BA2; set CPU_QR_CB_PRE_BA2=1 only for diagnostics.
        const bool cb_stage_ba2 = input.map && input.map->GetIniertialBA2();
        const bool cb_stage_ok = cb_stage_ba2 || CpuQrEnvFlag("CPU_QR_CB_PRE_BA2");
        if (cb_enable && !cb_stage_ok) {
            static std::atomic<int> s_cb_preba2_suppressed{0};
            const int nn = s_cb_preba2_suppressed.fetch_add(1);
            if (nn < 80) {
                std::cerr << "[CPU_QR_CIRCUIT_BREAK_SUPPRESS] seq=" << seq_num
                          << " ba2=0 -> pre-BA2 CB disabled" << std::endl;
            }
        }
        // Soft mode: on a trip, freeze only the IMU dynamic states (vel/bias) that blow up,
        // but still apply the visual pose+landmark refinement so the front-end keeps being
        // pulled back toward visual consistency. Hard mode (default) rejects the whole call.
        const bool cb_soft = CpuQrEnvFlag("CPU_QR_CB_SOFT");
        // Optional post-BA2 strict100 mode: judge the circuit breaker on the same
        // trust-region-limited deltas that will actually be written back.  Raw accel-bias
        // deltas can be large in healthy post-BA2 windows; hard-rejecting those before the
        // existing dynamic clamp starves visual BA and starts an FTL chain. Default is off
        // to preserve old experiments.
        const bool cb_use_applied_deltas = CpuQrEnvFlag("CPU_QR_CB_USE_APPLIED_DELTAS");
        const bool cb_ignore_ba = cb_stage_ba2 && CpuQrEnvFlag("CPU_QR_CB_IGNORE_BA_AFTER_BA2");
        const float cb_pose_dt = CpuQrEnvFloat("CPU_QR_CB_POSE_DT", 0.15f);
        const float cb_vel     = CpuQrEnvFloat("CPU_QR_CB_VEL", 0.5f);
        const float cb_ba      = CpuQrEnvFloat("CPU_QR_CB_BA", 0.06f);
        const float cb_bg      = CpuQrEnvFloat("CPU_QR_CB_BG", 0.01f);
        auto capped_norm_for_cb = [](float n, float cap) {
            return (cap > 0.0f && n > cap) ? cap : n;
        };
        auto dynamic_gain_for_cb = [&](int kid, bool velocity) {
            const bool is_anchor_kf = anchor_bias_freeze && anchor_kf_ids.count(kid) != 0;
            if (velocity) return is_anchor_kf ? (velocity_apply_gain * anchor_vel_gain) : velocity_apply_gain;
            return is_anchor_kf ? (bias_apply_gain * anchor_bias_gain) : bias_apply_gain;
        };
        bool cb_skip = false;
        if (cb_enable && cb_stage_ok) {
            float mx_pose = 0.0f, mx_vel = 0.0f, mx_ba = 0.0f, mx_bg = 0.0f;
            for (const auto& kv : pose_deltas) {
                float dt_norm = kv.second.head<3>().norm() * (cb_use_applied_deltas ? pose_apply_gain : 1.0f);
                if (cb_use_applied_deltas) dt_norm = capped_norm_for_cb(dt_norm, pose_damp_dt);
                mx_pose = std::max(mx_pose, dt_norm);
            }
            for (const auto& kv : velocity_deltas) {
                float n = kv.second.norm();
                if (cb_use_applied_deltas) {
                    n = capped_norm_for_cb(n, max_vel_total) * dynamic_gain_for_cb(kv.first, true);
                }
                mx_vel = std::max(mx_vel, n);
            }
            for (const auto& kv : acc_bias_deltas) {
                float n = kv.second.norm();
                if (cb_use_applied_deltas) {
                    n = capped_norm_for_cb(n, max_ba_total) * dynamic_gain_for_cb(kv.first, false);
                }
                mx_ba = std::max(mx_ba, n);
            }
            for (const auto& kv : gyro_bias_deltas) {
                float n = kv.second.norm();
                if (cb_use_applied_deltas) {
                    n = capped_norm_for_cb(n, max_bg_total) * dynamic_gain_for_cb(kv.first, false);
                }
                mx_bg = std::max(mx_bg, n);
            }
            const bool ba_trips = !cb_ignore_ba && mx_ba > cb_ba;
            if (mx_pose > cb_pose_dt || mx_vel > cb_vel || ba_trips || mx_bg > cb_bg) {
                cb_skip = true;
                std::cerr << "[CPU_QR_CIRCUIT_BREAK] seq=" << seq_num
                          << " mx_pose=" << mx_pose << " mx_vel=" << mx_vel
                          << " mx_ba=" << mx_ba << " mx_bg=" << mx_bg
                          << " mode=" << (cb_use_applied_deltas ? "applied" : "raw")
                          << " ignore_ba=" << (cb_ignore_ba ? 1 : 0)
                          << " (thr pose=" << cb_pose_dt << " vel=" << cb_vel
                          << " ba=" << cb_ba << " bg=" << cb_bg << ") -> "
                          << (cb_soft ? "freeze dynamic only (keep pose+lm)" : "reject whole call") << std::endl;
            }
        }

        for (KeyFrame* kf : input.local_kfs) {
            if (!kf || kf->isBad()) continue;
            if (cb_skip && !cb_soft) continue;  // hard mode: keep front-end state for all KFs
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
                } else if (per_pose_skip &&
                           (it->second.head<3>().norm() > per_pose_max_delta ||
                            it->second.tail<3>().norm() > per_pose_max_delta)) {
                    // Diverging single pose: keep its current tracking value, apply the rest.
                    int nn = s_pp_skip_log.fetch_add(1);
                    if (nn < 40 || verbose_calls) {
                        std::cerr << "[CPU_QR_PER_POSE_SKIP] seq=" << seq_num
                                  << " kf=" << kid
                                  << " dt_norm=" << it->second.head<3>().norm()
                                  << " omega_norm=" << it->second.tail<3>().norm()
                                  << " > " << per_pose_max_delta << std::endl;
                    }
                } else {
                    Eigen::Vector3f dt = it->second.head<3>() * pose_apply_gain;
                    Eigen::Vector3f omega = it->second.tail<3>() * pose_apply_gain;
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
            // Per-KF effective dynamic-state gains. Anchor candidates (oldest free KFs that
            // become the next window's fixed gauge) get the reduced anchor gain so their
            // weakly-observed fp32 vel/bias does not drift the gauge across calls.
            const bool is_anchor_kf = anchor_bias_freeze && anchor_kf_ids.count(kid) != 0;
            const float eff_vel_gain  = is_anchor_kf ? (velocity_apply_gain * anchor_vel_gain)  : velocity_apply_gain;
            const float eff_bias_gain = is_anchor_kf ? (bias_apply_gain     * anchor_bias_gain) : bias_apply_gain;
            if (is_anchor_kf) {
                std::cerr << "[CPU_QR_ANCHOR_FREEZE] seq=" << seq_num
                          << " kf=" << kid
                          << " eff_vel_gain=" << eff_vel_gain
                          << " eff_bias_gain=" << eff_bias_gain << std::endl;
            }
            if (!skip_velocity_apply && !cb_skip) {  // cb_skip in soft mode freezes dynamic states
                if (auto vit = velocity_deltas.find(kid); vit != velocity_deltas.end()) {
                    Eigen::Vector3f dv = clamp_dynamic_delta(vit->second, max_vel_total, "vel", kid);
                    result.velocity_updates[kf] = kf->GetVelocity() + eff_vel_gain * dv;
                }
            }
            if (!skip_bias_apply && !cb_skip) {  // cb_skip in soft mode freezes dynamic states
                auto bg_it = gyro_bias_deltas.find(kid);
                auto ba_it = acc_bias_deltas.find(kid);
                if (bg_it != gyro_bias_deltas.end() || ba_it != acc_bias_deltas.end()) {
                    IMU::Bias b = kf->GetImuBias();
                    Eigen::Vector3f bg(b.bwx, b.bwy, b.bwz);
                    Eigen::Vector3f ba(b.bax, b.bay, b.baz);
                    if (bg_it != gyro_bias_deltas.end()) {
                        Eigen::Vector3f dbg = clamp_dynamic_delta(bg_it->second, max_bg_total, "bg", kid);
                        bg += eff_bias_gain * dbg;
                    }
                    if (ba_it != acc_bias_deltas.end()) {
                        Eigen::Vector3f dba = clamp_dynamic_delta(ba_it->second, max_ba_total, "ba", kid);
                        ba += eff_bias_gain * dba;
                    }
                    result.bias_updates[kf] = IMU::Bias(ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z());
                }
            }
        }
        int lm_skipped = 0;
        int lm_reproj_skipped = 0;
        if (skip_all_landmarks || (cb_skip && !cb_soft)) {  // soft mode keeps visual landmarks
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

        // Post-optimization outlier rejection (matching g2o chi2 > 5.991 behavior).
        // Compute reprojection chi2 using updated pose+landmark; mark outliers for
        // caller to erase from map under mutex.
        const bool do_outlier_rejection = !CpuQrEnvFlag("CPU_QR_NO_OUTLIER_REJECT");
        if (do_outlier_rejection) {
            constexpr float chi2_mono_threshold = 5.991f;
            std::map<int, KeyFrame*> kf_by_id;
            for (KeyFrame* kf : input.local_kfs) if (kf) kf_by_id[static_cast<int>(kf->mnId)] = kf;
            for (KeyFrame* kf : input.fixed_kfs) if (kf) kf_by_id[static_cast<int>(kf->mnId)] = kf;
            std::map<int, MapPoint*> mp_by_id;
            for (MapPoint* mp : input.local_mps) if (mp && !mp->isBad()) mp_by_id[static_cast<int>(mp->mnId)] = mp;
            // Avoid duplicate (kf,mp) pairs
            std::set<std::pair<unsigned long, unsigned long>> seen;
            for (const DumpObservation& obs : payload.observations) {
                auto kf_it = kf_by_id.find(obs.pose_id);
                auto mp_it = mp_by_id.find(obs.landmark_id);
                if (kf_it == kf_by_id.end() || mp_it == mp_by_id.end()) continue;
                KeyFrame* kf = kf_it->second;
                MapPoint* mp = mp_it->second;
                auto key = std::make_pair(kf->mnId, mp->mnId);
                if (!seen.insert(key).second) continue;
                // Get updated pose
                Eigen::Matrix3f Rcw; Eigen::Vector3f tcw;
                auto pu = result.pose_updates.find(kf);
                if (pu != result.pose_updates.end()) {
                    Rcw = pu->second.rotationMatrix(); tcw = pu->second.translation();
                } else {
                    Sophus::SE3f T = kf->GetPose(); Rcw = T.rotationMatrix(); tcw = T.translation();
                }
                // Get updated landmark world position
                Eigen::Vector3f Pw;
                auto lu = result.landmark_updates.find(mp);
                if (lu != result.landmark_updates.end()) {
                    Pw = lu->second;
                } else {
                    Pw = mp->GetWorldPos().cast<float>();
                }
                // Project and compute chi2
                Eigen::Vector3f Pc = Rcw * Pw + tcw;
                if (Pc.z() <= 1e-6f) {
                    result.outliers_to_erase.emplace_back(kf, mp);
                    continue;
                }
                float inv_z = 1.0f / Pc.z();
                Eigen::Vector2f proj(obs.fx * Pc.x() * inv_z + obs.cx, obs.fy * Pc.y() * inv_z + obs.cy);
                Eigen::Vector2f err = proj - obs.pixel;
                float sigma2 = obs.sigma_pixel * obs.sigma_pixel;
                if (sigma2 < 1e-12f) sigma2 = 1.0f;
                float chi2 = err.squaredNorm() / sigma2;
                if (chi2 > chi2_mono_threshold) {
                    result.outliers_to_erase.emplace_back(kf, mp);
                }
            }
            if (verbose_calls || (!result.outliers_to_erase.empty() && seq_num < 50)) {
                std::cerr << "[CPU_QR_OUTLIER] seq=" << seq_num
                          << " total_obs=" << payload.observations.size()
                          << " outliers=" << result.outliers_to_erase.size() << std::endl;
            }
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
            // The strict full-replacement bootstrap must still provide enough visual
            // constraints. Per-KF ASIC feature caps are for steady-state hardware sizing;
            // applying them before IMU initialization starves the pure-visual BA
            // (e.g. 120 landmarks instead of the 400-landmark strict-good run).
            const bool pre_imu_bootstrap = input.map && !input.map->isImuInitialized();
            const bool uncap_obs_always = CpuQrEnvFlag("CPU_QR_UNCAP_OBS_ALWAYS");
            const bool post_ba2_uncap_obs = CpuQrEnvFlag("CPU_QR_UNCAP_OBS_AFTER_BA2") && input.map && input.map->GetIniertialBA2();
            const int obs_cap = (uncap_obs_always || pre_imu_bootstrap || post_ba2_uncap_obs) ? 2147483647 : ASIC_OBS_PER_LANDMARK_CAP;
            const int obs_keep = std::min(track, obs_cap);
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

        const bool pre_imu_bootstrap_window = input.map && !input.map->isImuInitialized();
        const bool uncap_obs_always_window = CpuQrEnvFlag("CPU_QR_UNCAP_OBS_ALWAYS");
        const bool post_ba2_uncap_window = CpuQrEnvFlag("CPU_QR_UNCAP_OBS_AFTER_BA2") && input.map && input.map->GetIniertialBA2();
        const int feature_cap = (uncap_obs_always_window || pre_imu_bootstrap_window || post_ba2_uncap_window) ? 2147483647 : ASIC_FEATURES_PER_KF_CAP;
        const int obs_cap_per_mp = (uncap_obs_always_window || pre_imu_bootstrap_window || post_ba2_uncap_window) ? 2147483647 : ASIC_OBS_PER_LANDMARK_CAP;
        std::map<KeyFrame*, int> kept_per_kf;
        std::map<MapPoint*, int> kept_per_mp;
        for (const CandidateObservation& co : candidates) {
            if (kept_per_kf[co.kf] >= feature_cap) continue;
            if (kept_per_mp[co.mp] >= obs_cap_per_mp) continue;
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


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


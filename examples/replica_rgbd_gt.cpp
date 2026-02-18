/**
 * replica_rgbd_gt.cpp
 * 
 * Photo-SLAM with Ground Truth camera poses from Replica dataset.
 * 
 * This executable runs ORB-SLAM3 normally (for feature extraction, keyframe
 * selection, and map point creation), but tells GaussianMapper to use GT poses
 * for all GaussianKeyframes instead of ORB-SLAM3's estimated poses.
 * 
 * Key difference from replica_rgbd.cpp:
 *   - Loads GT poses from traj.txt
 *   - Calls pGausMapper->setGTPoses() to inject GT into the mapping pipeline
 *   - ORB-SLAM3 tracking runs completely unmodified (no tracking failures)
 *   - Only the 3DGS reconstruction uses GT poses
 * 
 * Purpose: Isolate the mapping quality from tracking errors. If GT poses
 * produce significantly better reconstruction, it proves tracking is the
 * bottleneck.
 * 
 * Usage: Same as replica_rgbd (drop-in replacement)
 *   ./replica_rgbd_gt vocab orb_cfg gaus_cfg /path/to/sequence output_dir [no_viewer]
 */

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>

#include <opencv2/core/core.hpp>

#include "ORB-SLAM3/include/System.h"

#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

// ============================================================================
// GT Pose Loading
// ============================================================================

/**
 * Load ground truth poses from Replica traj.txt
 * Each line is a flattened 4x4 camera-to-world (c2w) matrix.
 * We apply the Replica convention (flip Y and Z axes) and convert to
 * ORB-SLAM3's world-to-camera (Tcw) format as Sophus::SE3f.
 */
std::vector<Sophus::SE3f> LoadGTPoses(const std::filesystem::path& traj_path)
{
    std::vector<Sophus::SE3f> poses;
    std::ifstream fin(traj_path);
    if (!fin.is_open()) {
        std::cerr << "[GT Poses] ERROR: Cannot open " << traj_path << std::endl;
        return poses;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Eigen::Matrix4f c2w;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                iss >> c2w(r, c);

        // Replica convention: flip Y and Z axes
        c2w.block<3,1>(0, 1) *= -1.0f;  // Y column
        c2w.block<3,1>(0, 2) *= -1.0f;  // Z column

        // Convert c2w -> w2c (Tcw) for ORB-SLAM3
        Eigen::Matrix4f w2c = c2w.inverse();
        Sophus::SE3f Tcw(w2c.block<3,3>(0,0), w2c.block<3,1>(0,3));
        poses.push_back(Tcw);
    }

    std::cout << "[GT Poses] Loaded " << poses.size() << " ground truth poses from "
              << traj_path << std::endl;
    return poses;
}

// ============================================================================
// Image Loading (same as replica_rgbd.cpp)
// ============================================================================

void LoadImages(const std::filesystem::path &pathImageDir, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD)
{
    for (const auto& imagePath : std::filesystem::directory_iterator(pathImageDir))
    {
        std::string name = imagePath.path().filename().string();
        if (name.rfind("frame", 0) == 0)
            vstrImageFilenamesRGB.push_back(imagePath.path().string());
        else if (name.rfind("depth", 0) == 0)
            vstrImageFilenamesD.push_back(imagePath.path().string());
        std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end());
        std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end());
    }
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }
    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_gaussian_mapping_settings"    /*3*/
                  << " path_to_sequence"                     /*4*/
                  << " path_to_trajectory_output_directory/" /*5*/
                  << " (optional)no_viewer"                  /*6*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 7)
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // ====================================================================
    // Load GT poses from traj.txt
    // ====================================================================
    std::string strSequenceDir = std::string(argv[4]);
    std::filesystem::path gt_traj_path = std::filesystem::path(strSequenceDir) / "traj.txt";
    std::vector<Sophus::SE3f> gt_poses = LoadGTPoses(gt_traj_path);
    if (gt_poses.empty()) {
        std::cerr << "[GT Poses] FATAL: No GT poses loaded. Aborting." << std::endl;
        return 1;
    }

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::filesystem::path pathImageDir(strSequenceDir);
    pathImageDir /= "results";
    LoadImages(pathImageDir, vstrImageFilenamesRGB, vstrImageFilenamesD);

    // Check consistency in the number of images
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }

    if ((int)gt_poses.size() < nImages) {
        std::cerr << "[GT Poses] WARNING: Only " << gt_poses.size()
                  << " GT poses for " << nImages << " images. "
                  << "Will use GT for first " << gt_poses.size() << " frames." << std::endl;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);

    // ====================================================================
    // INJECT GT POSES INTO GAUSSIAN MAPPER
    // ====================================================================
    // This tells GaussianMapper to use GT poses for all GaussianKeyframes
    // instead of ORB-SLAM3's estimated poses. ORB-SLAM3 still runs normally
    // for feature extraction, keyframe selection, and map point creation.
    pGausMapper->setGTPoses(gt_poses);

    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence with GT poses ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl;
    std::cout << "GT poses available: " << gt_poses.size() << std::endl << std::endl;

    // Main loop — ORB-SLAM3 runs completely unmodified
    cv::Mat imRGB, imD;
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;
        // Read image and depthmap from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(imRGB, imRGB, CV_BGR2RGB);
        imD = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = ni;

        if (imRGB.empty())
        {
            std::cerr << std::endl << "Failed to load image at: "
                      << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        if (imD.empty())
        {
            std::cerr << std::endl << "Failed to load image at: "
                      << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to ORB-SLAM3 — runs completely unmodified
        pSLAM->TrackRGBD(imRGB, imD, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Progress log every 200 frames
        if (ni % 200 == 0 || ni == nImages - 1) {
            std::cout << "[GT Poses] Frame " << ni << "/" << nImages << std::endl;
        }
    }

    // Stop all threads
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();

    // GPU peak usage
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Save camera trajectory
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    return 0;
}

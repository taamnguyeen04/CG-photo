/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper.h"

GaussianMapper::GaussianMapper(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // Mode
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Sensors
    switch (pSLAM->getSensorType())
    {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // Cameras
    // TODO: not only monocular
    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            // camera.width_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.width
            //                                              : graphics_utils::roundToIntegerMultipleOf16(
            //                                                    undistort_params.old_size_.width);
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

            // camera.height_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.height
            //                                               : graphics_utils::roundToIntegerMultipleOf16(
            //                                                     undistort_params.old_size_.height);
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();
    // Depth save scale: TUM=5000, Replica=6553.5
    if (settings_file["RGBD.depth_save_scale"].isReal())
        depth_save_scale_ = settings_file["RGBD.depth_save_scale"].operator float();
    else
        depth_save_scale_ = 5000.0f;  // Default: TUM format

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    // Wavelet Pyramid Configuration (optional, defaults to disabled)
    if (settings_file["WaveletPyramid.enabled"].operator int()) {
        wavelet_config_.enabled = true;
        
        // Parse edge detection method (sobel or wavelet)
        std::string edge_method_str = (std::string)settings_file["WaveletPyramid.edge_method"];
        if (edge_method_str == "sobel" || edge_method_str == "SOBEL") {
            wavelet_config_.use_sobel = true;
        } else {
            wavelet_config_.use_sobel = false;  // Default to wavelet
        }
        
        std::string wavelet_type_str = (std::string)settings_file["WaveletPyramid.type"];
        if (wavelet_type_str == "haar" || wavelet_type_str == "HAAR") {
            wavelet_config_.wavelet_type = wavelet::WaveletType::HAAR;
        } else if (wavelet_type_str == "db2" || wavelet_type_str == "DB2") {
            wavelet_config_.wavelet_type = wavelet::WaveletType::DB2;
        } else if (wavelet_type_str == "db4" || wavelet_type_str == "DB4") {
            wavelet_config_.wavelet_type = wavelet::WaveletType::DB4;
        }
        wavelet_config_.use_high_freq_loss = settings_file["WaveletPyramid.use_high_freq_loss"].operator int() != 0;
        wavelet_config_.high_freq_weight = settings_file["WaveletPyramid.high_freq_weight"].operator float();
        std::cout << "[Gaussian Mapper] Adaptive Densify ENABLED (method=" << edge_method_str 
                  << ", high_freq_loss=" << wavelet_config_.use_high_freq_loss 
                  << ", weight=" << wavelet_config_.high_freq_weight << ")" << std::endl;
    } else {
        wavelet_config_.enabled = false;
    }
    
    // Wavelet-guided Gaussian Initialization (independent of wavelet pyramid)
    if (settings_file["WaveletInit.enabled"].operator int()) {
        wavelet_config_.init_enabled = true;
        wavelet_config_.init_edge_threshold = settings_file["WaveletInit.edge_threshold"].operator float();
        wavelet_config_.init_max_extra_points = settings_file["WaveletInit.max_extra_points"].operator int();
        std::cout << "[Gaussian Mapper] Wavelet Init ENABLED (edge_threshold=" 
                  << wavelet_config_.init_edge_threshold 
                  << ", max_extra_points=" << wavelet_config_.init_max_extra_points << ")" << std::endl;
    } else {
        wavelet_config_.init_enabled = false;
    }

    // EFD: Error Frequency Decomposition for gradient boosting
    if (settings_file["EFD.enabled"].operator int()) {
        efd_config_.enabled = true;
        efd_config_.low_freq_weight  = settings_file["EFD.low_freq_weight"].operator float();
        efd_config_.high_freq_weight = settings_file["EFD.high_freq_weight"].operator float();
        efd_config_.boost_strength   = settings_file["EFD.boost_strength"].operator float();
        efd_config_.min_error_threshold = settings_file["EFD.min_error_threshold"].operator float();
        std::cout << "[Gaussian Mapper] EFD ENABLED (low=" << efd_config_.low_freq_weight
                  << ", high=" << efd_config_.high_freq_weight
                  << ", boost=" << efd_config_.boost_strength << ")" << std::endl;
    } else {
        efd_config_.enabled = false;
    }

    // MIG: Marginal Information Gain — two-phase transmittance-guided densification
    if (settings_file["MIG.enabled"].operator int()) {
        mig_config_.enabled = true;
        mig_config_.min_T_threshold     = settings_file["MIG.min_T_threshold"].operator float();
        mig_config_.min_error_threshold = settings_file["MIG.min_error_threshold"].operator float();
        mig_config_.budget_per_iter     = settings_file["MIG.budget_per_iter"].operator int();
        mig_config_.densify_interval    = settings_file["MIG.densify_interval"].operator int();
        mig_config_.lambda_T            = settings_file["MIG.lambda_T"].operator float();
        mig_config_.post_densify_window = settings_file["MIG.post_densify_window"].operator int();
        std::cout << "[Gaussian Mapper] MIG ENABLED (Phase1: lambda_T=" << mig_config_.lambda_T
                  << ", Phase2: budget=" << mig_config_.budget_per_iter
                  << ", window=" << mig_config_.post_densify_window << ")" << std::endl;
    } else {
        mig_config_.enabled = false;
    }

    // Fisher Information-based Uncertainty Configuration
    if (settings_file["Fisher.enabled"].operator int()) {
        fisher_config_.enabled = true;
        fisher_config_.prune_threshold = settings_file["Fisher.prune_threshold"].operator float();
        fisher_config_.update_interval = settings_file["Fisher.update_interval"].operator int();
        fisher_config_.prune_interval = settings_file["Fisher.prune_interval"].operator int();
        fisher_config_.ema_alpha = settings_file["Fisher.ema_alpha"].operator float();
        fisher_config_.min_observation_count = settings_file["Fisher.min_observations"].operator float();
        fisher_config_.weight_position = settings_file["Fisher.weight_position"].operator float();
        fisher_config_.weight_scale = settings_file["Fisher.weight_scale"].operator float();
        fisher_config_.weight_rotation = settings_file["Fisher.weight_rotation"].operator float();
        fisher_config_.weight_opacity = settings_file["Fisher.weight_opacity"].operator float();
        std::cout << "[Gaussian Mapper] Fisher Information ENABLED (prune_threshold=" 
                  << fisher_config_.prune_threshold 
                  << ", update_interval=" << fisher_config_.update_interval 
                  << ", prune_interval=" << fisher_config_.prune_interval << ")" << std::endl;
    } else {
        fisher_config_.enabled = false;
    }

    // CG-SLAM Uncertainty pruning configuration
    if (settings_file["Uncertainty.enable"].operator int()) {
        uncertainty_enabled_ = true;
        uncertainty_tau_ = settings_file["Uncertainty.tau"].operator float();
        uncertainty_prune_interval_ = settings_file["Uncertainty.prune_interval"].operator int();
        uncertainty_ema_alpha_ = settings_file["Uncertainty.ema_alpha"].operator float();
        std::cout << "[Gaussian Mapper] Uncertainty Pruning ENABLED (tau=" 
                  << uncertainty_tau_ 
                  << ", interval=" << uncertainty_prune_interval_ 
                  << ", ema_alpha=" << uncertainty_ema_alpha_ << ")" << std::endl;
    } else {
        uncertainty_enabled_ = false;
        std::cout << "[Gaussian Mapper] Uncertainty Pruning DISABLED" << std::endl;
    }

    // Guided Filter Dense Depth Initialization
    if (settings_file["GuidedDepth.enabled"].operator int()) {
        guided_depth_config_.enabled = true;
        guided_depth_config_.max_points_per_keyframe = settings_file["GuidedDepth.max_points_per_keyframe"].operator int();
        guided_depth_config_.edge_sample_ratio = settings_file["GuidedDepth.edge_sample_ratio"].operator float();
        guided_depth_config_.edge_threshold = settings_file["GuidedDepth.edge_threshold"].operator float();
        guided_depth_config_.depth_gradient_threshold = settings_file["GuidedDepth.depth_gradient_threshold"].operator float();
        guided_depth_config_.grid_cell_size = settings_file["GuidedDepth.grid_cell_size"].operator int();
        guided_depth_config_.min_valid_depth_ratio = settings_file["GuidedDepth.min_valid_depth_ratio"].operator float();
        std::cout << "[Gaussian Mapper] Guided Depth Dense Init ENABLED (max_pts=" 
                  << guided_depth_config_.max_points_per_keyframe 
                  << ", edge_ratio=" << guided_depth_config_.edge_sample_ratio
                  << ", grid=" << guided_depth_config_.grid_cell_size << ")" << std::endl;
    } else {
        guided_depth_config_.enabled = false;
    }

    // Geometry-Aware Gaussian Initialization
    if (settings_file["GeoAwareInit.enabled"].operator int()) {
        geo_aware_config_.enabled = true;
        geo_aware_config_.flatten_ratio = settings_file["GeoAwareInit.flatten_ratio"].operator float();
        geo_aware_config_.opacity_min = settings_file["GeoAwareInit.opacity_min"].operator float();
        geo_aware_config_.opacity_max = settings_file["GeoAwareInit.opacity_max"].operator float();
        geo_aware_config_.edge_opacity_threshold = settings_file["GeoAwareInit.edge_opacity_threshold"].operator float();
        geo_aware_config_.normal_confidence_threshold = settings_file["GeoAwareInit.normal_confidence_threshold"].operator float();
        std::cout << "[Gaussian Mapper] GeoAware Init ENABLED (flatten=" 
                  << geo_aware_config_.flatten_ratio 
                  << ", conf_th=" << geo_aware_config_.normal_confidence_threshold
                  << ", opacity=[" << geo_aware_config_.opacity_min
                  << "," << geo_aware_config_.opacity_max << "])" << std::endl;
    } else {
        geo_aware_config_.enabled = false;
    }

    // ConeGS: Cone-based Scale Init (replaces k-NN scaling)
    if (!settings_file["ConeScale.enabled"].empty() && settings_file["ConeScale.enabled"].operator int()) {
        cone_scale_config_.enabled = true;
        if (!settings_file["ConeScale.scale_multiplier"].empty())
            cone_scale_config_.scale_multiplier = settings_file["ConeScale.scale_multiplier"].operator float();
        if (!settings_file["ConeScale.min_scale"].empty())
            cone_scale_config_.min_scale = settings_file["ConeScale.min_scale"].operator float();
        if (!settings_file["ConeScale.max_scale"].empty())
            cone_scale_config_.max_scale = settings_file["ConeScale.max_scale"].operator float();
        std::cout << "[ConeGS] Cone Scale ENABLED (mult=" << cone_scale_config_.scale_multiplier
                  << ", min=" << cone_scale_config_.min_scale
                  << ", max=" << cone_scale_config_.max_scale << ")" << std::endl;
    }

    // ConeGS: Error-Guided Insertion (replaces Clone/Split)
    if (!settings_file["ConeDensify.enabled"].empty() && settings_file["ConeDensify.enabled"].operator int()) {
        cone_densify_config_.enabled = true;
        if (!settings_file["ConeDensify.budget_per_iter"].empty())
            cone_densify_config_.budget_per_iter = settings_file["ConeDensify.budget_per_iter"].operator int();
        if (!settings_file["ConeDensify.densify_interval"].empty())
            cone_densify_config_.densify_interval = settings_file["ConeDensify.densify_interval"].operator int();
        if (!settings_file["ConeDensify.prune_interval"].empty())
            cone_densify_config_.prune_interval = settings_file["ConeDensify.prune_interval"].operator int();
        if (!settings_file["ConeDensify.prune_opacity_threshold"].empty())
            cone_densify_config_.prune_opacity_threshold = settings_file["ConeDensify.prune_opacity_threshold"].operator float();
        if (!settings_file["ConeDensify.min_error_threshold"].empty())
            cone_densify_config_.min_error_threshold = settings_file["ConeDensify.min_error_threshold"].operator float();
        if (!settings_file["ConeDensify.flatten_ratio"].empty())
            cone_densify_config_.flatten_ratio = settings_file["ConeDensify.flatten_ratio"].operator float();
        std::cout << "[ConeGS] Error-Guided Insertion ENABLED (budget=" << cone_densify_config_.budget_per_iter
                  << ", interval=" << cone_densify_config_.densify_interval
                  << ", flatten=" << cone_densify_config_.flatten_ratio
                  << ", prune_th=" << cone_densify_config_.prune_opacity_threshold << ")" << std::endl;
    }

    // Gradient-based Clone/Split toggle
    if (!settings_file["Densification.gradient_based"].empty())
        gradient_densify_enabled_ = settings_file["Densification.gradient_based"].operator int();
    std::cout << "[Densification] Gradient-based Clone/Split: " << (gradient_densify_enabled_ ? "ENABLED" : "DISABLED") << std::endl;

    // ConeGS: Pre-activation Opacity Penalty (replaces Opacity Reset)
    if (!settings_file["Optimization.lambda_opacity_penalty"].empty())
        lambda_opacity_penalty_ = settings_file["Optimization.lambda_opacity_penalty"].operator float();
    if (lambda_opacity_penalty_ > 0.0f) {
        std::cout << "[ConeGS] Opacity Penalty ENABLED (lambda=" << lambda_opacity_penalty_ << ")" << std::endl;
    }

    // Depth Back-Projection Initialization (stride-based)
    if (!settings_file["DepthBackproject.enabled"].empty() && settings_file["DepthBackproject.enabled"].operator int()) {
        depth_backproject_config_.enabled = true;
        if (!settings_file["DepthBackproject.mode"].empty())
            depth_backproject_config_.mode = settings_file["DepthBackproject.mode"].operator int();
        if (!settings_file["DepthBackproject.stride"].empty())
            depth_backproject_config_.stride = settings_file["DepthBackproject.stride"].operator int();
        if (!settings_file["DepthBackproject.stride_min"].empty())
            depth_backproject_config_.stride_min = settings_file["DepthBackproject.stride_min"].operator int();
        if (!settings_file["DepthBackproject.stride_max"].empty())
            depth_backproject_config_.stride_max = settings_file["DepthBackproject.stride_max"].operator int();
        if (!settings_file["DepthBackproject.block_size"].empty())
            depth_backproject_config_.block_size = settings_file["DepthBackproject.block_size"].operator int();
        if (!settings_file["DepthBackproject.edge_threshold"].empty())
            depth_backproject_config_.edge_threshold = settings_file["DepthBackproject.edge_threshold"].operator float();
        if (!settings_file["DepthBackproject.edge_ratio"].empty())
            depth_backproject_config_.edge_ratio = settings_file["DepthBackproject.edge_ratio"].operator float();
        if (!settings_file["DepthBackproject.max_points_per_keyframe"].empty())
            depth_backproject_config_.max_points_per_keyframe = settings_file["DepthBackproject.max_points_per_keyframe"].operator int();
        if (!settings_file["DepthBackproject.min_depth"].empty())
            depth_backproject_config_.min_depth = settings_file["DepthBackproject.min_depth"].operator float();
        if (!settings_file["DepthBackproject.max_depth"].empty())
            depth_backproject_config_.max_depth = settings_file["DepthBackproject.max_depth"].operator float();
        // Guided Filter
        if (!settings_file["DepthBackproject.guided_filter"].empty())
            depth_backproject_config_.guided_filter_enabled = settings_file["DepthBackproject.guided_filter"].operator int();
        if (!settings_file["DepthBackproject.guided_filter_radius"].empty())
            depth_backproject_config_.guided_filter_radius = settings_file["DepthBackproject.guided_filter_radius"].operator int();
        if (!settings_file["DepthBackproject.guided_filter_eps"].empty())
            depth_backproject_config_.guided_filter_eps = settings_file["DepthBackproject.guided_filter_eps"].operator float();
        // Depth gradient filter
        if (!settings_file["DepthBackproject.depth_grad_filter"].empty())
            depth_backproject_config_.depth_grad_filter = settings_file["DepthBackproject.depth_grad_filter"].operator int();
        if (!settings_file["DepthBackproject.depth_grad_threshold"].empty())
            depth_backproject_config_.depth_grad_threshold = settings_file["DepthBackproject.depth_grad_threshold"].operator float();
        if (depth_backproject_config_.mode == 1) {
            std::cout << "[Gaussian Mapper] Depth Back-Projection ENABLED (ADAPTIVE"
                      << ", stride=[" << depth_backproject_config_.stride_min
                      << "," << depth_backproject_config_.stride_max << "]"
                      << ", block=" << depth_backproject_config_.block_size
                      << ", edge_th=" << depth_backproject_config_.edge_threshold
                      << ", ratio=" << depth_backproject_config_.edge_ratio
                      << ", budget=" << depth_backproject_config_.max_points_per_keyframe
                      << ", depth=[" << depth_backproject_config_.min_depth
                      << "," << depth_backproject_config_.max_depth << "]";
            if (depth_backproject_config_.guided_filter_enabled)
                std::cout << ", GuidedFilter(r=" << depth_backproject_config_.guided_filter_radius
                          << ",eps=" << depth_backproject_config_.guided_filter_eps << ")";
            if (depth_backproject_config_.depth_grad_filter)
                std::cout << ", DepthGradFilter(th=" << depth_backproject_config_.depth_grad_threshold << ")";
            std::cout << ")" << std::endl;
        } else {
            std::cout << "[Gaussian Mapper] Depth Back-Projection ENABLED (UNIFORM"
                      << ", stride=" << depth_backproject_config_.stride
                      << ", depth=[" << depth_backproject_config_.min_depth
                      << "," << depth_backproject_config_.max_depth << "])" << std::endl;
        }
    } else {
        depth_backproject_config_.enabled = false;
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    // Depth-Photo-SLAM: Loss weights (with defaults if not in config)
    if (!settings_file["Optimization.lambda_geo"].empty())
        opt_params_.lambda_geo_ = settings_file["Optimization.lambda_geo"].operator float();
    if (!settings_file["Optimization.lambda_smooth"].empty())
        opt_params_.lambda_smooth_ = settings_file["Optimization.lambda_smooth"].operator float();
    if (!settings_file["Optimization.lambda_var"].empty())
        opt_params_.lambda_var_ = settings_file["Optimization.lambda_var"].operator float();
    if (!settings_file["Optimization.lambda_iso"].empty())
        opt_params_.lambda_iso_ = settings_file["Optimization.lambda_iso"].operator float();
    if (!settings_file["Optimization.lambda_align"].empty())
        opt_params_.lambda_align_ = settings_file["Optimization.lambda_align"].operator float();
    if (!settings_file["Optimization.lambda_reg"].empty())
        opt_params_.lambda_reg_ = settings_file["Optimization.lambda_reg"].operator float();

    if (!settings_file["Optimization.jmvo_enabled"].empty())
        opt_params_.jmvo_enabled_ = settings_file["Optimization.jmvo_enabled"].operator int() != 0;
        
    if (!settings_file["Optimization.lambda_esc"].empty())
        opt_params_.lambda_esc_ = settings_file["Optimization.lambda_esc"].operator float();
    if (!settings_file["Optimization.lambda_g1"].empty())
        opt_params_.lambda_g1_ = settings_file["Optimization.lambda_g1"].operator float();
    if (!settings_file["Optimization.lambda_g2"].empty())
        opt_params_.lambda_g2_ = settings_file["Optimization.lambda_g2"].operator float();
    if (!settings_file["Optimization.gradient_log_interval"].empty())
        opt_params_.gradient_log_interval_ = settings_file["Optimization.gradient_log_interval"].operator int();

    // ====================================================================
    // Molding-GS: TSDF-Anchored Gaussian Splatting Configuration
    // ====================================================================
    if (!settings_file["TSDF.enabled"].empty())
        tsdf_config_.enabled = settings_file["TSDF.enabled"].operator int() != 0;
    if (!settings_file["TSDF.voxel_size"].empty())
        tsdf_config_.voxel_size = settings_file["TSDF.voxel_size"].operator float();
    if (!settings_file["TSDF.truncation"].empty())
        tsdf_config_.truncation = settings_file["TSDF.truncation"].operator float();
    if (!settings_file["TSDF.max_radius"].empty())
        tsdf_config_.max_radius = settings_file["TSDF.max_radius"].operator float();
    if (!settings_file["TSDF.min_weight"].empty())
        tsdf_config_.min_weight = settings_file["TSDF.min_weight"].operator float();
    if (!settings_file["TSDF.lambda_sdf"].empty())
        tsdf_config_.lambda_sdf = settings_file["TSDF.lambda_sdf"].operator float();
    if (!settings_file["TSDF.lambda_normal"].empty())
        tsdf_config_.lambda_normal = settings_file["TSDF.lambda_normal"].operator float();
    if (!settings_file["TSDF.strain_threshold"].empty())
        tsdf_config_.strain_threshold = settings_file["TSDF.strain_threshold"].operator float();
    if (!settings_file["TSDF.prune_sdf_threshold"].empty())
        tsdf_config_.prune_sdf_threshold = settings_file["TSDF.prune_sdf_threshold"].operator float();
    if (!settings_file["TSDF.prune_interval"].empty())
        tsdf_config_.prune_interval = settings_file["TSDF.prune_interval"].operator int();
    if (!settings_file["TSDF.fusion_interval"].empty())
        tsdf_config_.fusion_interval = settings_file["TSDF.fusion_interval"].operator int();

    if (tsdf_config_.enabled) {
        std::cout << "[Molding-GS] TSDF-Anchored 3DGS ENABLED" << std::endl;
        std::cout << "  voxel_size=" << tsdf_config_.voxel_size
                  << " truncation=" << tsdf_config_.truncation
                  << " max_radius=" << tsdf_config_.max_radius << std::endl;
        std::cout << "  lambda_sdf=" << tsdf_config_.lambda_sdf
                  << " lambda_normal=" << tsdf_config_.lambda_normal
                  << " strain_th=" << tsdf_config_.strain_threshold << std::endl;
    }

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapper::run()
{
    // First loop: Initial gaussian mapping
    while (!isStopped()) {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            pSLAM_->getAtlas()->clearMappingOperation();

            // Get initial sparse map
            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vpKFs = pMap->GetAllKeyFrames();
                vpMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vpMPs){
                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);
                    scene_->cachePoint3D(pMP->mnId, point3D);
                }
                for (const auto& pKF : vpKFs){
                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        // Camera
                        Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        cv::Mat imgRGB = pKF->imgLeftRGB;
                        if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        // Auxiliary Image
                        cv::Mat imgAux = pKF->imgAuxiliary;
                        if (this->sensor_type_ == RGBD)
                            camera.undistortImage(imgAux, imgAux_undistorted);
                        else
                            imgAux_undistorted = imgAux;

                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        new_kf->img_filename_ = pKF->mNameFile;
                        new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                        new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                        new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }
                    new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // Features
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                }
            }

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        if (wavelet_config_.enabled) {
                            // Wavelet Pyramid: Use Haar decomposition for better edge preservation
                            pkf->gaus_pyramid_original_image_[l] = wavelet::waveletDownsample(
                                img_gpu,
                                pkf->gaus_pyramid_height_[l],
                                pkf->gaus_pyramid_width_[l],
                                torch::kCUDA);
                        } else {
                            // Gaussian Pyramid: Standard bilinear downsampling
                            cv::cuda::GpuMat img_resized;
                            cv::cuda::resize(img_gpu, img_resized,
                                            cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                            pkf->gaus_pyramid_original_image_[l] =
                                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                        }
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        if (wavelet_config_.enabled) {
                            // Wavelet Pyramid: Use Haar decomposition for better edge preservation
                            pkf->gaus_pyramid_original_image_[l] = wavelet::waveletDownsampleCPU(
                                pkf->img_undist_,
                                pkf->gaus_pyramid_height_[l],
                                pkf->gaus_pyramid_width_[l],
                                device_type_);
                        } else {
                            // Gaussian Pyramid: Standard bilinear downsampling
                            cv::Mat img_resized;
                            cv::resize(pkf->img_undist_, img_resized,
                                    cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                            pkf->gaus_pyramid_original_image_[l] =
                                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                        }
                    }
                }
            }

            // Prepare for training
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }

            // Molding-GS: Initialize Local TSDF volume
            if (tsdf_config_.enabled) {
                local_tsdf_ = std::make_unique<local_tsdf::LocalTSDF>(
                    tsdf_config_.voxel_size, tsdf_config_.truncation);
                std::cout << "[Molding-GS] LocalTSDF initialized: voxel="
                          << tsdf_config_.voxel_size << "m, trunc="
                          << tsdf_config_.truncation << "m" << std::endl;
            }

            // Invoke training once
            trainForOneIteration();

            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
            break;
        }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();

        if (pSLAM_->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

    // Third loop: Tail gaussian optimization
    // Guarantee we run AT LEAST until densify_until_iter + buffer,
    // so that densification always completes even if SLAM shut down early
    // (e.g. when TSDF / heavy CPU modules slow down the mapping loop).
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    int min_tail_iters = std::max(
        opt_params_.densify_until_iter_ + 2 * densify_interval,
        SLAM_stop_iter + n_delay_iters
    );
    if (getIteration() < min_tail_iters) {
        std::cout << "[Tail Opt] SLAM stopped at iter=" << SLAM_stop_iter
                  << ", but densify_until=" << opt_params_.densify_until_iter_
                  << ". Continuing to iter=" << min_tail_iters << " ..." << std::endl;
    }
    while (getIteration() < min_tail_iters
           || getIteration() - SLAM_stop_iter <= n_delay_iters
           || getIteration() % densify_interval <= n_delay_iters
           || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    // Render depth for all trajectory poses (not just keyframes)
    std::filesystem::path traj_file = result_dir_ / "CameraTrajectory_TUM.txt";
    if (std::filesystem::exists(traj_file)) {
        renderAndRecordAllTrajectoryPoses(traj_file, "_shutdown");
    }
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    
    // Save Gaussian count to file for pipeline script extraction
    {
        std::ofstream count_file(result_dir_ / "gaussian_count.txt");
        if (count_file.is_open()) {
            count_file << gaussians_->xyz_.size(0) << std::endl;
            count_file.close();
        }
    }
    
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

void GaussianMapper::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Depth-Photo-SLAM: Prepare ground truth depth for L_geo (RGBD only)
    torch::Tensor gt_depth;
    bool has_gt_depth = false;
    
    if (this->sensor_type_ == RGBD && viewpoint_cam->img_auxiliary_undist_.data) {
        // Convert depth image (CV_32FC1 in meters) to tensor
        cv::Mat depth_undist = viewpoint_cam->img_auxiliary_undist_;
        
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat depth_gpu;
            depth_gpu.upload(depth_undist);
            gt_depth = tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu);
        } else {
            gt_depth = tensor_utils::cvMat2TorchTensor_Float32(depth_undist, device_type_);
        }
        
        // Resize if using Gaussian pyramid training (match rendered resolution)
        if (training_level != num_gaus_pyramid_sub_levels_ || 
            gt_depth.size(1) != image_height || gt_depth.size(2) != image_width) {
            
            // gt_depth shape: [1, H, W] or [C, H, W]
            if (gt_depth.dim() == 3 && gt_depth.size(0) == 3) {
                // Take only first channel if RGB depth map
                gt_depth = gt_depth.index({0}).unsqueeze(0);
            }
            if (gt_depth.dim() == 2) {
                gt_depth = gt_depth.unsqueeze(0);
            }
            
            gt_depth = torch::nn::functional::interpolate(
                gt_depth.unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{image_height, image_width})
                    .mode(torch::kNearest)
            ).squeeze(0).squeeze(0);
        } else {
            // Squeeze to [H, W]
            if (gt_depth.dim() == 3) {
                if (gt_depth.size(0) == 3) {
                    gt_depth = gt_depth.index({0});
                } else {
                    gt_depth = gt_depth.squeeze(0);
                }
            }
        }
        
        has_gt_depth = true;
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);
    // Depth-Photo-SLAM: Extract depth outputs
    auto depth = std::get<4>(render_pkg);
    auto depth_sq = std::get<5>(render_pkg);
    auto median_depth = std::get<6>(render_pkg);
    // CG-SLAM: Extract uncertainty output for L_var
    auto uncertainty = std::get<7>(render_pkg);
    // MIG: Extract transmittance map T[H, W]
    auto T_map = std::get<8>(render_pkg);
    // ESC: Extract per-Gaussian cov2D and view-space depths
    auto primary_cov2D = std::get<9>(render_pkg);       // [P, 3] (xx, xy, yy)
    auto primary_view_depths = std::get<10>(render_pkg); // [P]

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    torch::Tensor Ll1;
    
    // EFD Loss Weighting: frequency-aware per-pixel L1 weight
    // Wavelet decomposes |rendered-gt| → high-freq error regions get higher loss weight
    // Unlike gradient boosting, this works WITH Adam optimizer (not against it)
    if (efd_config_.enabled &&
        getIteration() >= opt_params_.densify_from_iter_ &&
        getIteration() < opt_params_.densify_until_iter_) {
        auto efd_weight_map = wavelet::efdComputeWeightMap(
            masked_image.detach(), gt_image.detach(), efd_config_);  // [H, W] range [1, boost_strength]
        auto efd_weight_3d = efd_weight_map.unsqueeze(0);  // [1, H, W] → broadcasts with [3, H, W]
        Ll1 = ((masked_image - gt_image).abs() * efd_weight_3d).mean();
    }
    // MIG Phase 1: T-weighted L1 — boost loss at under-covered pixels
    else if (mig_config_.enabled && mig_config_.lambda_T > 0.0f &&
        getIteration() > opt_params_.densify_until_iter_) {
        auto T_weight = (1.0f + mig_config_.lambda_T *
                         T_map.detach().clamp(0.0f, 1.0f)).unsqueeze(0);  // [1,H,W]
        Ll1 = ((masked_image - gt_image).abs() * T_weight).mean();
    } else {
        Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    }
    float lambda_dssim = lambdaDssim();
    
    // Read loss weights from config (early to enable conditional computation)
    float lambda_geo = opt_params_.lambda_geo_;
    float lambda_align = opt_params_.lambda_align_;
    float lambda_var = opt_params_.lambda_var_;
    float lambda_iso = opt_params_.lambda_iso_;
    float lambda_smooth = opt_params_.lambda_smooth_;
    float lambda_reg = opt_params_.lambda_reg_;
    float lambda_g1 = opt_params_.lambda_g1_;
    float lambda_g2 = opt_params_.lambda_g2_;
    
    // Depth-Photo-SLAM: Compute depth-aware losses (ONLY if lambda > 0)
    torch::Tensor L_align = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_var = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_geo = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_smooth = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_iso = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_reg = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_g1 = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_g2 = torch::zeros(1, torch::TensorOptions().device(device_type_));
    
    // L_align: Alignment loss (only if enabled)
    if (lambda_align > 0.0f) {
        L_align = loss_utils::depth_alignment_loss(depth, median_depth);
    }
    
    // L_var: Variance Loss - CG-SLAM Eq. 10-11 (only if enabled AND has GT depth)
    // Paper formula: U = Σ αᵢTᵢ(dᵢ - D)² 
    // Since rasterizer doesn't receive gt_depth, we compute U = (rendered_depth - gt_depth)² here
    // L_var = (1/HW) Σ |U_n|
    if (lambda_var > 0.0f && has_gt_depth) {
        // Create valid depth mask
        auto depth_valid_mask = (gt_depth > RGBD_min_depth_) & (gt_depth < RGBD_max_depth_);
        depth_valid_mask = depth_valid_mask & mask.squeeze().to(torch::kBool);
        
        // CG-SLAM Eq. 10: U = (d - D)² where d = rendered depth, D = GT depth
        auto depth_squeezed = depth.squeeze();  // [H, W] rendered depth
        auto depth_error = depth_squeezed - gt_depth;
        auto uncertainty_map = depth_error * depth_error;  // U = (d - D)²
        
        // Apply mask and compute mean (Eq. 11: L_var = (1/HW) Σ |U_n|)
        auto masked_uncertainty = uncertainty_map * depth_valid_mask.to(uncertainty_map.dtype());
        auto num_valid = depth_valid_mask.sum().clamp_min(1.0f);
        L_var = masked_uncertainty.sum() / num_valid;
        
        // DEBUG: Log L_var statistics every 100 iterations
        if (getIteration() % 100 == 0) {
            auto U_masked = uncertainty_map.masked_select(depth_valid_mask);
            if (U_masked.numel() > 0) {
                float U_mean = U_masked.mean().item<float>();
                float U_max = U_masked.max().item<float>();
                float U_min = U_masked.min().item<float>();
                std::cout << "[CG-SLAM L_var] iter=" << getIteration() 
                          << " | U=(d-D)²: min=" << U_min << " max=" << U_max << " mean=" << U_mean
                          << " | L_var=" << L_var.item<float>()
                          << " | valid_pixels=" << U_masked.numel() << std::endl;
            }
        }
    }
    
    // L_iso: Isotropy Loss (only if enabled) - prevents needle-like Gaussians
    if (lambda_iso > 0.0f) {
        L_iso = loss_utils::isotropy_loss(gaussians_->scaling_, /*epsilon=*/1.5f, /*use_log_scales=*/true);
    }
    
    // L_geo: Geometric sensor depth loss (only if enabled AND has GT depth)
    if (lambda_geo > 0.0f && has_gt_depth) {
        // Create valid depth mask: exclude invalid sensor readings
        auto depth_valid_mask = (gt_depth > RGBD_min_depth_) & (gt_depth < RGBD_max_depth_);
        
        // Apply undistortion mask as well
        depth_valid_mask = depth_valid_mask & mask.squeeze().to(torch::kBool);
        
        // Squeeze depth to [H, W] for loss computation
        auto depth_squeezed = depth.squeeze();
        
        // Compute L_geo using sensor_depth_loss
        L_geo = loss_utils::sensor_depth_loss(
            depth_squeezed,       // Rendered depth [H, W]
            gt_depth,             // Ground truth depth [H, W]
            depth_valid_mask      // Valid pixel mask [H, W]
        );
    }
    
    // L_smooth: Edge-aware Smoothness Loss (only if enabled)
    if (lambda_smooth > 0.0f) {
        L_smooth = loss_utils::smoothness_loss(depth, gt_image);
    }
    
    // L_reg: Planar Regularization (MonoGS++) - encourages flat Gaussians on surfaces
    if (lambda_reg > 0.0f) {
        L_reg = loss_utils::planar_regularization_loss(gaussians_->scaling_, /*min_scale_floor=*/0.01f, /*use_log_scales=*/true);
    }
    
    // L_g1: First-order gradient loss (2D-3DGS) - preserves edges
    if (lambda_g1 > 0.0f) {
        L_g1 = loss_utils::first_order_gradient_loss(masked_image, gt_image, device_type_);
    }
    
    // L_g2: Second-order gradient loss (2D-3DGS) - preserves corners/curvature
    if (lambda_g2 > 0.0f) {
        L_g2 = loss_utils::second_order_gradient_loss(masked_image, gt_image, device_type_);
    }
    
    // L_wavelet: Wavelet edge loss (optional) - preserves high-frequency details
    torch::Tensor L_wavelet = torch::zeros({1}, torch::TensorOptions().device(device_type_));
    if (wavelet_config_.enabled && wavelet_config_.use_high_freq_loss) {
        L_wavelet = wavelet::waveletEdgeLoss(
            masked_image, gt_image,
            wavelet_config_.high_freq_weight,   // LH weight
            wavelet_config_.high_freq_weight,   // HL weight
            wavelet_config_.high_freq_weight * 0.5f);  // HH weight (less important)
    }
    
    // ConeGS: Pre-activation opacity penalty (replaces Opacity Reset)
    // L_o = lambda_o × mean(o_pre) — ép Gaussian rác giảm opacity dần
    torch::Tensor L_opacity = torch::zeros({1}, torch::TensorOptions().device(device_type_));
    if (lambda_opacity_penalty_ > 0.0f) {
        L_opacity = gaussians_->opacity_.mean();  // Raw pre-activation (pre-sigmoid) value
    }

    // ====================================================================
    // Molding-GS: TSDF Anchor Loss + Normal Alignment Loss
    // Sync: wait for background fusion to finish (non-blocking test) before
    // querying the volume. If still running, skip this iteration's TSDF loss.
    // ====================================================================
    bool tsdf_ready = !tsdf_fusion_future_.valid() ||
                      tsdf_fusion_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    torch::Tensor L_sdf = torch::zeros(1, torch::TensorOptions().device(device_type_));
    torch::Tensor L_normal = torch::zeros(1, torch::TensorOptions().device(device_type_));
    
    // TSDF Loss: only compute every N iterations to avoid blocking GPU.
    // querySDF/queryGradient/queryWeight are CPU hash lookups over 200k+ Gaussians
    // and 2M+ voxels — calling them every iteration freezes training for minutes.
    const int tsdf_loss_interval = 10;  // Compute TSDF loss every 10 iters
    if (tsdf_config_.enabled && local_tsdf_ && local_tsdf_->numVoxels() > 0 &&
        tsdf_ready && (getIteration() % tsdf_loss_interval == 0)) {
        auto xyz = gaussians_->getXYZ();  // [N, 3] — differentiable
        
        // Batch query TSDF: SDF values, gradients (normals), observation weights
        auto sdf_vals = local_tsdf_->querySDF(xyz);       // [N] on CUDA
        auto sdf_grads = local_tsdf_->queryGradient(xyz);  // [N, 3] on CUDA
        auto sdf_weights = local_tsdf_->queryWeight(xyz);  // [N] on CUDA
        
        // L_sdf: Pull Gaussian centers to SDF=0 surface
        L_sdf = loss_utils::sdf_anchor_loss(
            sdf_vals, gaussians_->opacity_, sdf_weights, tsdf_config_.min_weight);
        
        // L_normal: Align Gaussian Z-axis to TSDF surface normal
        L_normal = loss_utils::normal_alignment_loss(
            gaussians_->rotation_, sdf_grads, sdf_weights, tsdf_config_.min_weight);
        
        // Log every 500 iterations
        if (getIteration() % 500 == 0) {
            auto observed = (sdf_weights > tsdf_config_.min_weight);
            int64_t n_observed = observed.sum().item<int64_t>();
            float mean_sdf = 0.0f;
            if (n_observed > 0) {
                mean_sdf = sdf_vals.abs().masked_select(observed).mean().item<float>();
            }
            std::cout << "[Molding-GS] iter=" << getIteration()
                      << " | L_sdf=" << L_sdf.item<float>()
                      << " L_normal=" << L_normal.item<float>()
                      << " | observed=" << n_observed << "/" << xyz.size(0)
                      << " | mean|SDF|=" << std::fixed << std::setprecision(4) << mean_sdf
                      << " | voxels=" << local_tsdf_->numVoxels()
                      << std::endl;
        }
    }

    // Combine losses with weights
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_))
                + lambda_geo * L_geo
                + lambda_align * L_align
                + lambda_var * L_var
                + lambda_smooth * L_smooth
                + lambda_iso * L_iso
                + lambda_reg * L_reg
                + lambda_g1 * L_g1
                + lambda_g2 * L_g2
                + L_wavelet  // Wavelet edge loss
                + lambda_opacity_penalty_ * L_opacity  // ConeGS opacity penalty
                + tsdf_config_.lambda_sdf * L_sdf      // Molding-GS: SDF anchor
                + tsdf_config_.lambda_normal * L_normal; // Molding-GS: normal alignment

    // ========================================================================
    // Analytical ESC: Epipolar Scale Consistency (no secondary rendering)
    // Computes Σ2D analytically from Gaussian params + camera poses.
    // Fully differentiable — gradients flow to xyz, scales, rotations.
    // ========================================================================
    if (opt_params_.jmvo_enabled_ && opt_params_.lambda_esc_ > 0.0f && scene_->keyframes().size() > 3) {
        // Find a neighbor keyframe via temporal interleaving
        auto it = scene_->keyframes().find(viewpoint_cam->fid_);
        std::shared_ptr<GaussianKeyframe> neighbor_cam = nullptr;

        if (getIteration() % 2 == 0) {
            if (it != scene_->keyframes().begin())
                neighbor_cam = std::prev(it)->second;
            else {
                auto next_it = std::next(it);
                if (next_it != scene_->keyframes().end())
                    neighbor_cam = next_it->second;
            }
        } else {
            auto next_it = std::next(it);
            if (next_it != scene_->keyframes().end())
                neighbor_cam = next_it->second;
            else if (it != scene_->keyframes().begin())
                neighbor_cam = std::prev(it)->second;
        }

        if (neighbor_cam) {
            // Compute focal lengths from FoV
            float fx_A = viewpoint_cam->image_width_ / (2.0f * std::tan(viewpoint_cam->FoVx_ * 0.5f));
            float fy_A = viewpoint_cam->image_height_ / (2.0f * std::tan(viewpoint_cam->FoVy_ * 0.5f));
            float fx_B = neighbor_cam->image_width_ / (2.0f * std::tan(neighbor_cam->FoVx_ * 0.5f));
            float fy_B = neighbor_cam->image_height_ / (2.0f * std::tan(neighbor_cam->FoVy_ * 0.5f));

            // Analytical ESC — no rendering needed, pure PyTorch matrix ops
            auto L_esc = loss_utils::analytical_esc_loss(
                gaussians_->getXYZ(),                    // [N, 3] positions
                gaussians_->getScalingActivation(),      // [N, 3] activated scales
                gaussians_->getRotationActivation(),     // [N, 4] normalized quaternions
                viewpoint_cam->world_view_transform_,    // [4, 4] primary view matrix
                neighbor_cam->world_view_transform_,     // [4, 4] neighbor view matrix
                fx_A, fy_A, fx_B, fy_B,
                radii,                                   // [N] primary visibility
                radii                                    // [N] use primary radii as proxy (neighbor not rendered)
            );

            loss = loss + opt_params_.lambda_esc_ * L_esc;
        }
    }
    // ========================================================================
    // END Analytical ESC
    // ========================================================================
    
    // ========================================================================
    // LOSS COMPONENT LOGGING - Log individual loss values for visualization
    // ========================================================================
    const int loss_log_interval = 100;  // Log every 100 iterations
    static std::ofstream loss_log_file;
    static bool loss_log_initialized = false;
    
    if (loss_log_interval > 0 && !loss_log_initialized && getIteration() == 1) {
        auto loss_log_path = result_dir_ / "loss_components.csv";
        loss_log_file.open(loss_log_path);
        loss_log_file << "iteration,L1,DSSIM,L_geo,L_var,L_smooth,L_iso,L_reg,total" << std::endl;
        loss_log_initialized = true;
    }
    
    if (loss_log_interval > 0 && getIteration() % loss_log_interval == 0 && getIteration() > 0) {
        auto L_dssim = 1.0 - loss_utils::ssim(masked_image, gt_image, device_type_);
        float l1_val = Ll1.item<float>();
        float dssim_val = L_dssim.item<float>();
        float geo_val = L_geo.item<float>();
        float var_val = L_var.item<float>();
        float smooth_val = L_smooth.item<float>();
        float iso_val = L_iso.item<float>();
        float reg_val = L_reg.item<float>();
        float total_val = loss.item<float>();
        
        if (loss_log_file.is_open()) {
            loss_log_file << getIteration() << "," 
                         << l1_val << "," << dssim_val << "," 
                         << geo_val << "," << var_val << "," << smooth_val << ","
                         << iso_val << "," << reg_val << ","
                         << total_val << std::endl;
        }
    }
    
    // ========================================================================
    // GRADIENT CONFLICT ANALYSIS - Log gradient directions for each loss
    // ========================================================================
    const int gradient_log_interval = opt_params_.gradient_log_interval_;  // From YAML config
    
    if (gradient_log_interval > 0 && getIteration() % gradient_log_interval == 0 && getIteration() > 0) {
        // ====== DEBUG: Check tensor properties ======
        std::cout << "\n[DEBUG] Tensor requires_grad status:" << std::endl;
        std::cout << "  depth.requires_grad = " << (depth.requires_grad() ? "true" : "false") << std::endl;
        std::cout << "  depth_sq.requires_grad = " << (depth_sq.requires_grad() ? "true" : "false") << std::endl;
        std::cout << "  median_depth.requires_grad = " << (median_depth.requires_grad() ? "true" : "false") << std::endl;
        std::cout << "  gaussians_->xyz_.requires_grad = " << (gaussians_->xyz_.requires_grad() ? "true" : "false") << std::endl;
        std::cout << "  gaussians_->scaling_.requires_grad = " << (gaussians_->scaling_.requires_grad() ? "true" : "false") << std::endl;
        
        // Debug tensor values
        std::cout << "[DEBUG] Tensor values:" << std::endl;
        std::cout << "  depth mean = " << depth.mean().item<float>() << std::endl;
        std::cout << "  depth_sq mean = " << depth_sq.mean().item<float>() << std::endl;
        std::cout << "  L_var value = " << L_var.item<float>() << std::endl;
        std::cout << "  L_geo value = " << L_geo.item<float>() << std::endl;
        std::cout << "  L_align value = " << L_align.item<float>() << std::endl;
        std::cout << "  L_smooth value = " << L_smooth.item<float>() << std::endl;
        std::cout << "  L_iso value = " << L_iso.item<float>() << std::endl;
        std::cout << "  L_reg value = " << L_reg.item<float>() << std::endl;
        std::cout << "  L_g1 value = " << L_g1.item<float>() << std::endl;
        std::cout << "  L_g2 value = " << L_g2.item<float>() << std::endl;
        // ====== END DEBUG ======
        
        // Compute individual weighted losses
        auto loss_l1 = (1.0 - lambda_dssim) * Ll1;
        auto loss_dssim = lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
        auto loss_geo = lambda_geo * L_geo;
        auto loss_align = lambda_align * L_align;
        auto loss_var = lambda_var * L_var;
        auto loss_smooth = lambda_smooth * L_smooth;
        auto loss_iso = lambda_iso * L_iso;
        auto loss_reg = lambda_reg * L_reg;
        auto loss_g1 = lambda_g1 * L_g1;
        auto loss_g2 = lambda_g2 * L_g2;
        
        // Store gradients for xyz parameter
        std::vector<torch::Tensor> grads_xyz;
        std::vector<torch::Tensor> grads_scaling;  // For L_iso, L_reg
        std::vector<std::string> names = {"L1", "DSSIM", "L_geo", "L_align", "L_var", "L_smooth", "L_iso", "L_reg", "L_g1", "L_g2"};
        std::vector<torch::Tensor> losses_vec = {loss_l1, loss_dssim, loss_geo, loss_align, loss_var, loss_smooth, loss_iso, loss_reg, loss_g1, loss_g2};
        
        for (size_t i = 0; i < losses_vec.size(); ++i) {
            // Zero gradients
            if (gaussians_->xyz_.grad().defined()) {
                gaussians_->xyz_.grad().zero_();
            }
            if (gaussians_->scaling_.grad().defined()) {
                gaussians_->scaling_.grad().zero_();
            }
            
            // Backward for individual loss (skip if no grad_fn, e.g., when lambda=0)
            if (losses_vec[i].requires_grad() && losses_vec[i].grad_fn()) {
                losses_vec[i].backward(/*gradient=*/{}, /*retain_graph=*/true);
            }
            
            // Store xyz gradient
            if (gaussians_->xyz_.grad().defined()) {
                grads_xyz.push_back(gaussians_->xyz_.grad().flatten().clone());
            } else {
                grads_xyz.push_back(torch::zeros({gaussians_->xyz_.numel()}, gaussians_->xyz_.options()));
            }
            
            // Store scaling gradient (for L_iso debug)
            if (gaussians_->scaling_.grad().defined()) {
                grads_scaling.push_back(gaussians_->scaling_.grad().flatten().clone());
            } else {
                grads_scaling.push_back(torch::zeros({gaussians_->scaling_.numel()}, gaussians_->scaling_.options()));
            }
        }
        
        // Compute pairwise cosine similarities
        std::cout << "\n[Gradient Conflict Analysis] Iteration " << getIteration() << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        for (size_t i = 0; i < grads_xyz.size(); ++i) {
            for (size_t j = i + 1; j < grads_xyz.size(); ++j) {
                float norm_i = grads_xyz[i].norm().item<float>();
                float norm_j = grads_xyz[j].norm().item<float>();
                
                float cos_sim = 0.0f;
                if (norm_i > 1e-8f && norm_j > 1e-8f) {
                    cos_sim = torch::dot(grads_xyz[i], grads_xyz[j]).item<float>() / (norm_i * norm_j);
                }
                
                std::string status;
                if (cos_sim < -0.1f) status = "CONFLICT";
                else if (cos_sim < 0.3f) status = "WEAK";
                else status = "ALIGNED";
                
                std::cout << "  " << names[i] << " vs " << names[j] 
                          << ": cos=" << std::fixed << std::setprecision(3) << cos_sim
                          << " [" << status << "]" << std::endl;
            }
        }
        
        // Log gradient magnitudes for xyz
        std::cout << "Gradient magnitudes (xyz_):" << std::endl;
        for (size_t i = 0; i < grads_xyz.size(); ++i) {
            std::cout << "  " << names[i] << ": " << std::fixed << std::setprecision(6) 
                      << grads_xyz[i].norm().item<float>() << std::endl;
        }
        
        // Log gradient magnitudes for scaling (for L_iso debug)
        std::cout << "Gradient magnitudes (scaling_):" << std::endl;
        for (size_t i = 0; i < grads_scaling.size(); ++i) {
            std::cout << "  " << names[i] << ": " << std::fixed << std::setprecision(6) 
                      << grads_scaling[i].norm().item<float>() << std::endl;
        }
        std::cout << "----------------------------------------\n" << std::endl;
        
        // Zero gradients for actual training
        if (gaussians_->xyz_.grad().defined()) {
            gaussians_->xyz_.grad().zero_();
        }
        if (gaussians_->scaling_.grad().defined()) {
            gaussians_->scaling_.grad().zero_();
        }
    }
    // ========================================================================
    // END GRADIENT CONFLICT ANALYSIS
    // ========================================================================
    
    loss.backward();

    torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, depth, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_, result_dir_);


        // Densification
        if (getIteration() < opt_params_.densify_until_iter_) {

            // Standard: Keep track of max radii in image-space for pruning
            // (EFD now operates via loss weighting, not gradient boosting)
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if (gradient_densify_enabled_ &&
                (getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                
                // Wavelet-based Adaptive Densification
                float adaptive_grad_threshold = densifyGradThreshold();
                
                if (wavelet_config_.enabled && has_gt_depth) {
                    float mean_edge_strength = 0.5f;
                    
                    if (wavelet_config_.use_sobel) {
                        auto gray = gt_image.mean(0, true);
                        auto sobel_x = torch::tensor({{-1.0f, 0.0f, 1.0f},
                                                       {-2.0f, 0.0f, 2.0f},
                                                       {-1.0f, 0.0f, 1.0f}}, 
                                                      torch::TensorOptions().device(gt_image.device()));
                        auto sobel_y = torch::tensor({{-1.0f, -2.0f, -1.0f},
                                                       { 0.0f,  0.0f,  0.0f},
                                                       { 1.0f,  2.0f,  1.0f}}, 
                                                      torch::TensorOptions().device(gt_image.device()));
                        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0);
                        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0);
                        auto gray_batch = gray.unsqueeze(0);
                        auto grad_x = torch::nn::functional::conv2d(gray_batch, sobel_x, 
                            torch::nn::functional::Conv2dFuncOptions().padding(1));
                        auto grad_y = torch::nn::functional::conv2d(gray_batch, sobel_y,
                            torch::nn::functional::Conv2dFuncOptions().padding(1));
                        auto grad_mag = torch::sqrt(grad_x * grad_x + grad_y * grad_y);
                        auto edge_max = grad_mag.max();
                        auto edge_min = grad_mag.min();
                        if ((edge_max - edge_min).item<float>() > 1e-6f) {
                            grad_mag = (grad_mag - edge_min) / (edge_max - edge_min + 1e-6f);
                        }
                        mean_edge_strength = grad_mag.mean().item<float>();
                        if (mean_edge_strength > 0.15f) adaptive_grad_threshold *= 0.5f;
                        else if (mean_edge_strength < 0.05f) adaptive_grad_threshold *= 1.5f;
                    } else {
                        auto wavelet_decomp = wavelet::haarDecompose2D(gt_image);
                        auto edge_map = wavelet_decomp.LH.abs() + wavelet_decomp.HL.abs() + wavelet_decomp.HH.abs();
                        auto edge_max = edge_map.max();
                        auto edge_min = edge_map.min();
                        if ((edge_max - edge_min).item<float>() > 1e-6f) {
                            edge_map = (edge_map - edge_min) / (edge_max - edge_min + 1e-6f);
                        }
                        mean_edge_strength = edge_map.mean().item<float>();
                        if (mean_edge_strength > 0.6f) adaptive_grad_threshold *= 0.5f;
                        else if (mean_edge_strength < 0.3f) adaptive_grad_threshold *= 1.5f;
                    }
                }
                
                gaussians_->densifyAndPrune(
                    adaptive_grad_threshold,
                    densify_min_opacity_,
                    scene_->cameras_extent_,
                    size_threshold
                );
            }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();

            // ====================================================================
            // Molding-GS: Strain-Energy Densification (Physics-based Split)
            // Splits Gaussians that have high strain (far from TSDF surface)
            // using tangent-plane projection for split direction
            // ====================================================================
            if (tsdf_config_.enabled && local_tsdf_ && local_tsdf_->numVoxels() > 0 &&
                gradient_densify_enabled_ &&
                (getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval() == 0) &&
                (!tsdf_fusion_future_.valid() || tsdf_fusion_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
                
                torch::NoGradGuard no_grad_strain;
                
                auto xyz = gaussians_->getXYZ().detach();              // [N, 3]
                auto sdf_vals = local_tsdf_->querySDF(xyz);             // [N]
                auto sdf_grads = local_tsdf_->queryGradient(xyz);       // [N, 3]
                auto sdf_weights = local_tsdf_->queryWeight(xyz);       // [N]
                auto scales_act = gaussians_->getScalingActivation().detach(); // [N, 3]
                
                // Strain energy: E = |SDF(μ)|
                auto strain = sdf_vals.abs();
                auto observed = sdf_weights > tsdf_config_.min_weight;
                auto max_scale = std::get<0>(scales_act.max(1));
                auto large_enough = max_scale > 0.005f;  // At least 5mm
                
                // Split candidates: high strain + observed + large enough
                auto split_mask = (strain > tsdf_config_.strain_threshold) & observed & large_enough;
                int64_t n_split = split_mask.sum().item<int64_t>();
                
                // Limit splits to avoid explosion, but don't restrict too heavily
                int64_t max_splits = std::max<int64_t>(500, static_cast<int64_t>(xyz.size(0) * 0.05));
                if (n_split > max_splits) {
                    auto split_indices_all = split_mask.nonzero().squeeze(1);
                    auto strain_at_candidates = strain.index({split_indices_all});
                    auto [top_strain, top_idx] = strain_at_candidates.topk(max_splits);
                    split_mask = torch::zeros_like(split_mask);
                    split_mask.index_put_({split_indices_all.index({top_idx})}, true);
                    n_split = max_splits;
                }
                
                if (n_split > 0) {
                    auto split_idx = split_mask.nonzero().squeeze(1);  // [K]
                    auto split_xyz = xyz.index({split_idx});           // [K, 3]
                    auto split_sdf = sdf_vals.index({split_idx});      // [K]
                    auto split_grad = sdf_grads.index({split_idx});    // [K, 3] normals
                    auto split_scales = gaussians_->scaling_.index({split_idx}).detach(); // [K, 3] log-scale
                    auto split_rotations = gaussians_->rotation_.index({split_idx}).detach(); // [K, 4]
                    auto split_features_dc = gaussians_->features_dc_.index({split_idx}).detach();
                    auto split_features_rest = gaussians_->features_rest_.index({split_idx}).detach();
                    auto split_opacity = gaussians_->opacity_.index({split_idx}).detach();
                    
                    auto split_scales_act = scales_act.index({split_idx}); // [K, 3]
                    
                    // Find longest axis per Gaussian
                    auto [max_s, max_dim] = split_scales_act.max(1);  // [K], [K]
                    
                    // Build v_long: unit vector along longest axis in world frame
                    // For simplicity, use the column of rotation matrix corresponding to max_dim
                    auto qw = split_rotations.index({"...", 0});
                    auto qx = split_rotations.index({"...", 1});
                    auto qy = split_rotations.index({"...", 2});
                    auto qz = split_rotations.index({"...", 3});
                    
                    // Full rotation matrix columns
                    auto col0 = torch::stack({
                        1.f - 2.f*(qy*qy + qz*qz),
                        2.f*(qx*qy + qw*qz),
                        2.f*(qx*qz - qw*qy)}, 1);  // [K, 3]
                    auto col1 = torch::stack({
                        2.f*(qx*qy - qw*qz),
                        1.f - 2.f*(qx*qx + qz*qz),
                        2.f*(qy*qz + qw*qx)}, 1);  // [K, 3]
                    auto col2 = torch::stack({
                        2.f*(qx*qz + qw*qy),
                        2.f*(qy*qz - qw*qx),
                        1.f - 2.f*(qx*qx + qy*qy)}, 1);  // [K, 3]
                    
                    // Select v_long based on max_dim
                    auto v_long = torch::where(
                        (max_dim == 0).unsqueeze(1), col0,
                        torch::where((max_dim == 1).unsqueeze(1), col1, col2)); // [K, 3]
                    
                    // Project v_long onto tangent plane: v_split = v_long - (v_long·n)n
                    auto n = split_grad;  // [K, 3] surface normal
                    auto dot_vn = (v_long * n).sum(1, true);  // [K, 1]
                    auto v_split = v_long - dot_vn * n;       // [K, 3]
                    
                    // Handle collinear case (v_long ∥ n): use random tangent vector
                    auto v_split_norm = v_split.norm(2, 1, true).clamp_min(1e-8f);  // [K, 1]
                    auto is_collinear = (v_split_norm.squeeze(1) < 1e-4f);  // [K]
                    
                    if (is_collinear.any().item<bool>()) {
                        // Generate random tangent vector for collinear cases
                        auto random_vec = torch::randn({n_split, 3}, xyz.options());
                        auto random_proj = random_vec - (random_vec * n).sum(1, true) * n;
                        auto random_norm = random_proj.norm(2, 1, true).clamp_min(1e-8f);
                        random_proj = random_proj / random_norm;
                        v_split = torch::where(is_collinear.unsqueeze(1), random_proj, v_split);
                        v_split_norm = v_split.norm(2, 1, true).clamp_min(1e-8f);
                    }
                    
                    v_split = v_split / v_split_norm;  // Normalize
                    
                    // Displacement: quarter of the scale along the split direction (more conservative)
                    auto displacement = 0.25f * max_s.unsqueeze(1) * v_split;  // [K, 3]
                    
                    // Create 2 children displaced along v_split
                    auto child1_xyz = split_xyz + displacement;
                    auto child2_xyz = split_xyz - displacement;
                    auto new_xyz = torch::cat({child1_xyz, child2_xyz}, 0);  // [2K, 3]
                    
                    // Scale conservation: child scale = parent / sqrt(2)
                    auto new_scales = (split_scales + std::log(1.0f / std::sqrt(2.0f)))
                                      .repeat({2, 1});  // [2K, 3]
                    
                    // Inherit rotation, features, opacity from parent
                    auto new_rotations = split_rotations.repeat({2, 1});
                    auto new_features_dc = split_features_dc.repeat({2, 1, 1});
                    auto new_features_rest = split_features_rest.repeat({2, 1, 1});
                    auto new_opacity = split_opacity.repeat({2, 1});
                    auto new_exist_since = torch::ones(
                        2 * n_split,
                        torch::TensorOptions().dtype(torch::kInt32).device(device_type_)) * getIteration();
                    
                    // MUST REMOVE PARENTS FIRST! 
                    // prunePoints uses split_mask which has size N.
                    // If we add children first, the size becomes N + 2K, causing an IndexError.
                    gaussians_->prunePoints(split_mask);

                    // Add children (appends 2K new points to the back)
                    gaussians_->densificationPostfix(
                        new_xyz, new_features_dc, new_features_rest,
                        new_opacity, new_scales, new_rotations, new_exist_since);
                    
                    if (getIteration() % 500 == 0) {
                        std::cout << "[Molding-GS Strain Split] iter=" << getIteration()
                                  << " | split=" << n_split
                                  << " | total=" << gaussians_->xyz_.size(0) << std::endl;
                    }
                }
            }

            // ====================================================================
            // Molding-GS: SDF-based Pruning (kill floaters)
            // Removes Gaussians that are far from any observed surface
            // ====================================================================
            if (tsdf_config_.enabled && local_tsdf_ && local_tsdf_->numVoxels() > 0 &&
                getIteration() % tsdf_config_.prune_interval == 0 &&
                getIteration() > opt_params_.densify_from_iter_ &&
                (!tsdf_fusion_future_.valid() || tsdf_fusion_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
                
                torch::NoGradGuard no_grad_prune;
                
                auto xyz = gaussians_->getXYZ().detach();
                auto sdf_vals = local_tsdf_->querySDF(xyz);
                auto sdf_weights = local_tsdf_->queryWeight(xyz);
                
                // Prune Gaussians with large |SDF| in well-observed regions
                auto prune_mask = (sdf_vals.abs() > tsdf_config_.prune_sdf_threshold) &
                                  (sdf_weights > tsdf_config_.min_weight);
                int num_pruned = prune_mask.sum().item<int>();
                
                if (num_pruned > 0) {
                    // Cap pruning at 5% per iteration
                    int max_prune = static_cast<int>(gaussians_->xyz_.size(0) * 0.05f);
                    if (num_pruned > max_prune) {
                        auto prune_indices_all = prune_mask.nonzero().squeeze(1);
                        auto sdf_at_candidates = sdf_vals.abs().index({prune_indices_all});
                        auto [top_sdf, top_idx] = sdf_at_candidates.topk(max_prune);
                        prune_mask = torch::zeros_like(prune_mask);
                        prune_mask.index_put_({prune_indices_all.index({top_idx})}, true);
                        num_pruned = max_prune;
                    }
                    
                    gaussians_->prunePoints(prune_mask);
                    
                    if (getIteration() % 500 == 0 || num_pruned > 100) {
                        std::cout << "[Molding-GS SDF Prune] iter=" << getIteration()
                                  << " | pruned=" << num_pruned
                                  << " | remaining=" << gaussians_->xyz_.size(0) << std::endl;
                    }
                }
            }

            // ====================================================================
            // HYBRID: ConeGS Error-Guided Insertion (supplement to Clone/Split)
            // Adds extra Gaussians at high-error regions on top of Clone/Split
            // ====================================================================
            if (cone_densify_config_.enabled &&
                (getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % cone_densify_config_.densify_interval == 0)) {
                
                // 1. Error-weighted pixel sampling modulated by GVS (Depth Variance)
                auto sampled_pixels = conegs::errorWeightedSample(
                    masked_image.detach(), gt_image.detach(), gt_depth.detach(),
                    cone_densify_config_.budget_per_iter,
                    cone_densify_config_.min_error_threshold);
                
                if (sampled_pixels.size(0) > 0 && has_gt_depth) {
                    // 2. Get camera intrinsics and pose
                    Camera& cam_cone = scene_->cameras_.at(viewpoint_cam->camera_id_);
                    float fx = cam_cone.params_[0];
                    float fy = cam_cone.params_[1];
                    float cx = cam_cone.params_[2];
                    float cy = cam_cone.params_[3];
                    
                    // Camera-to-world transform
                    Sophus::SE3f Tcw_se3 = viewpoint_cam->getPosef();
                    Eigen::Matrix4f Twc_eigen = Tcw_se3.inverse().matrix();
                    torch::Tensor Twc = tensor_utils::EigenMatrix2TorchTensor(
                        Twc_eigen, device_type_);
                    
                    // 3. Backproject sampled pixels to 3D using sensor depth
                    auto [new_xyz, new_colors, new_depths, valid_indices] = 
                        conegs::backprojectPixels(
                            sampled_pixels, gt_depth,
                            gt_image, fx, fy, cx, cy, Twc);
                    
                    if (new_xyz.size(0) > 0) {
                        // 4. Compute cone-based scales
                        torch::Tensor new_scales;
                        if (cone_scale_config_.enabled) {
                            new_scales = conegs::computeConeScale(
                                new_depths, fx, fy, cone_scale_config_);
                        } else {
                            auto dist2 = distCUDA2(new_xyz);
                            dist2 = torch::clamp_min(dist2, 0.0000001f);
                            new_scales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});
                        }
                        
                        // 4.5. Advanced ConeGS: Normal-Aligned Surfels (NAS)
                        // Extract valid pixel locations
                        auto u_all = sampled_pixels.index({torch::indexing::Slice(), 0});
                        auto v_all = sampled_pixels.index({torch::indexing::Slice(), 1});
                        auto u_valid = u_all.index({valid_indices});
                        auto v_valid = v_all.index({valid_indices});

                        // Compute Surface Normal from Depth Map and convert to World Space Rotation Quaternion
                        auto new_rotations = conegs::computeNormalsAndRotations(
                            gt_depth, u_valid, v_valid, new_depths, fx, fy, Twc);

                        // Morphological Squashing: flatten the Gaussian along its local Z-axis (normal)
                        // flatten_ratio controls thinness: 0.05=20x thin, 0.3=3.3x thin, 1.0=sphere
                        auto z_scales = new_scales.select(1, 2);
                        z_scales = z_scales + std::log(cone_densify_config_.flatten_ratio);
                        new_scales.select(1, 2).copy_(z_scales);
                        
                        // 5. Initialize features and opacity
                        auto new_sh_colors = (new_colors - 0.5f) / 0.28209479177387814f;
                        auto new_features_dc = new_sh_colors.unsqueeze(1);  // [N, 1, 3]
                        int64_t n_new = new_xyz.size(0);
                        int64_t n_rest_sh = gaussians_->features_rest_.size(1);
                        std::vector<int64_t> rest_size = {n_new, n_rest_sh, 3};
                        auto new_features_rest = torch::zeros(
                            at::IntArrayRef(rest_size),
                            torch::TensorOptions().device(device_type_));
                        float logit_05 = 0.0f;  // sigmoid(0) = 0.5 (logit 0.5 is 0)
                        std::vector<int64_t> opa_size = {n_new, 1};
                        auto new_opacities = torch::ones(
                            at::IntArrayRef(opa_size),
                            torch::TensorOptions().device(device_type_)) * logit_05;
                        
                        // 6. Add to model
                        auto new_exist_since = torch::ones(
                            n_new,
                            torch::TensorOptions().dtype(torch::kInt32).device(device_type_)) * getIteration();
                        gaussians_->densificationPostfix(
                            new_xyz, new_features_dc, new_features_rest,
                            new_opacities, new_scales, new_rotations, new_exist_since);
                        
                        if (getIteration() % 500 == 0) {
                            std::cout << "[ConeGS Hybrid] iter=" << getIteration()
                                      << " | inserted=" << new_xyz.size(0)
                                      << " | total=" << gaussians_->xyz_.size(0) << std::endl;
                        }
                    }
                }
            }
        }
        
        // ==================================================================
        // MIG Phase 2: Gap-Filling Insertion (after densification window)
        // Insert new Gaussians at remaining under-covered high-error pixels
        // Only runs AFTER Clone/Split has finished to avoid competition
        // ==================================================================
        if (mig_config_.enabled &&
            has_gt_depth &&
            (getIteration() > opt_params_.densify_until_iter_) &&
            (getIteration() < opt_params_.densify_until_iter_ + mig_config_.post_densify_window) &&
            (getIteration() % mig_config_.densify_interval == 0)) {

            torch::NoGradGuard no_grad_mig;

            // Safety: clamp T_map to [0,1]
            auto T_map_safe = T_map.detach().clamp(0.0f, 1.0f);
            if (T_map_safe.is_cuda() && T_map_safe.numel() > 0) {

            // BUG FIX: mask BOTH rendered and GT → avoid artificial error at undistort borders
            auto masked_gt_mig = gt_image.detach() * mask.detach();  // [3, H, W]
            auto error_map = (masked_image.detach() - masked_gt_mig).abs().mean(0);  // [H, W]

            // Also exclude mask==0 pixels (border) from valid candidates
            auto mask_2d = mask.detach().squeeze(0);  // [H, W] — undistored area mask
            auto mig_score = T_map_safe * error_map;  // [H, W]

            // Mask: T > threshold AND error > threshold AND inside valid image area
            auto valid_mask = (T_map_safe > mig_config_.min_T_threshold) &
                              (error_map > mig_config_.min_error_threshold) &
                              (mask_2d > 0.5f);

            // Get valid pixel indices
            auto valid_indices = valid_mask.nonzero().contiguous();  // [N_valid, 2] — (row, col)
            int64_t n_valid = valid_indices.size(0);

            if (n_valid > 0) {
                // Sample top-K pixels by MIG score
                auto valid_scores = mig_score.index({
                    valid_indices.index({torch::indexing::Slice(), 0}),
                    valid_indices.index({torch::indexing::Slice(), 1})});  // [N_valid]
                int64_t budget = std::min((int64_t)mig_config_.budget_per_iter, n_valid);
                auto [top_scores_mig, top_idx_mig] = valid_scores.topk(budget, 0, true, false);
                auto sampled_rc = valid_indices.index({top_idx_mig});  // [budget, 2]

                // Get camera intrinsics
                Camera& cam_mig = scene_->cameras_.at(viewpoint_cam->camera_id_);
                float fx_mig = cam_mig.params_[0];
                float fy_mig = cam_mig.params_[1];
                float cx_mig = cam_mig.params_[2];
                float cy_mig = cam_mig.params_[3];

                // Camera-to-world transform
                Sophus::SE3f Tcw_se3_mig = viewpoint_cam->getPosef();
                Eigen::Matrix4f Twc_eigen_mig = Tcw_se3_mig.inverse().matrix();
                torch::Tensor Twc_mig = tensor_utils::EigenMatrix2TorchTensor(Twc_eigen_mig, device_type_);

                // Sample depth at selected pixels
                auto rows_mig = sampled_rc.index({torch::indexing::Slice(), 0}).contiguous();
                auto cols_mig = sampled_rc.index({torch::indexing::Slice(), 1}).contiguous();
                auto sampled_depth_mig = gt_depth.index({rows_mig, cols_mig});  // [budget]
                auto depth_valid_mig = (sampled_depth_mig > RGBD_min_depth_) & (sampled_depth_mig < RGBD_max_depth_);
                auto valid_rows_mig  = rows_mig.index({depth_valid_mig}).contiguous();
                auto valid_cols_mig  = cols_mig.index({depth_valid_mig}).contiguous();
                auto valid_depth_mig = sampled_depth_mig.index({depth_valid_mig}).contiguous();

                int64_t n_insert_mig = valid_rows_mig.size(0);
                if (n_insert_mig > 0) {
                    // Unproject to 3D: X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d
                    auto u_mig = valid_cols_mig.to(torch::kFloat32);
                    auto v_mig = valid_rows_mig.to(torch::kFloat32);
                    auto Xc_mig = (u_mig - cx_mig) * valid_depth_mig / fx_mig;
                    auto Yc_mig = (v_mig - cy_mig) * valid_depth_mig / fy_mig;
                    auto Zc_mig = valid_depth_mig;
                    auto pts_cam_mig = torch::stack({Xc_mig, Yc_mig, Zc_mig, torch::ones_like(Zc_mig)}, 1);
                    auto new_xyz_h_mig = torch::mm(Twc_mig, pts_cam_mig.t());  // [4, N]
                    auto new_xyz_mig = new_xyz_h_mig.index({torch::indexing::Slice(0, 3),
                                                             torch::indexing::Slice()}).t().contiguous();  // [N, 3]

                    // Safety: check for NaN/Inf positions — skip if any found
                    auto pos_valid = torch::isfinite(new_xyz_mig).all(1);  // [N]
                    if (pos_valid.any().item<bool>()) {
                        new_xyz_mig = new_xyz_mig.index({pos_valid}).contiguous();
                        valid_rows_mig = valid_rows_mig.index({pos_valid}).contiguous();
                        valid_cols_mig = valid_cols_mig.index({pos_valid}).contiguous();
                        valid_depth_mig = valid_depth_mig.index({pos_valid}).contiguous();
                        n_insert_mig = new_xyz_mig.size(0);
                    } else {
                        n_insert_mig = 0;
                    }

                    if (n_insert_mig > 0) {
                    // Sample colors from GT image [3, H, W] → [N, 3]
                    auto new_colors_mig = gt_image.index({torch::indexing::Slice(), valid_rows_mig, valid_cols_mig})
                                                  .permute({1, 0}).contiguous();

                    // FIX: Use depth-proportional scale instead of distCUDA2
                    // distCUDA2 crashes with small point clouds (<8 points)
                    // Depth-proportional: scale ∝ pixel_size_in_world = depth / focal_length
                    float avg_focal = 0.5f * (fx_mig + fy_mig);
                    auto pixel_scale = valid_depth_mig / avg_focal;  // [N] — world-space pixel size
                    auto new_scales_mig = torch::log(pixel_scale).unsqueeze(-1).repeat({1, 3});  // [N, 3]

                    // Initialize SH features from RGB (RGB2SH conversion)
                    auto new_sh_mig = (new_colors_mig - 0.5f) / 0.28209479177387814f;
                    auto new_features_dc_mig = new_sh_mig.unsqueeze(1);  // [N, 1, 3]
                    int64_t n_rest_sh_mig = gaussians_->features_rest_.size(1);
                    auto new_features_rest_mig = torch::zeros(
                        {n_insert_mig, n_rest_sh_mig, 3},
                        torch::TensorOptions().device(device_type_));

                    // FIX: Use sigmoid(0.5) ≈ logit(0.5) = 0 as initial opacity
                    // logit(0.1) = -2.2 was too transparent → scattered light hurt existing Gaussians
                    float logit_05_mig = 0.0f;  // sigmoid(0) = 0.5
                    auto new_opacities_mig = torch::ones(
                        {n_insert_mig, 1},
                        torch::TensorOptions().device(device_type_)) * logit_05_mig;
                    auto new_rotations_mig = torch::zeros(
                        {n_insert_mig, 4},
                        torch::TensorOptions().device(device_type_));
                    new_rotations_mig.index_put_({torch::indexing::Slice(), 0}, 1.0f);

                    auto new_exist_since_mig = torch::ones(
                        n_insert_mig,
                        torch::TensorOptions().dtype(torch::kInt32).device(device_type_)) * getIteration();
                    gaussians_->densificationPostfix(
                        new_xyz_mig, new_features_dc_mig, new_features_rest_mig,
                        new_opacities_mig, new_scales_mig, new_rotations_mig, new_exist_since_mig);

                    if (getIteration() % 500 == 0) {
                        std::cout << "[MIG] iter=" << getIteration()
                                  << " | inserted=" << n_insert_mig
                                  << " | total=" << gaussians_->xyz_.size(0) << std::endl;
                    }
                    } // end n_insert_mig > 0 (after NaN filter)
                }
            }
            } // end T_map_safe guard
        }

        // CG-SLAM: Uncertainty-based pruning (Eq. 13 approximation)
        // Paper: νᵢ = (1/(M₁+...+Mₖ)) Σ Σ αᵢᵏ·ᵖ Tᵢᵏ·ᵖ (Dₚᵏ - dᵢᵏ)²
        // Only runs when Uncertainty.enable: 1 in YAML config
        if (uncertainty_enabled_ &&
            getIteration() > opt_params_.densify_until_iter_ && 
            getIteration() % uncertainty_prune_interval_ == 0 &&
            has_gt_depth) {
            
            int64_t num_gaussians = gaussians_->xyz_.size(0);
            
            // Initialize per-Gaussian uncertainty tensor
            auto per_gaussian_uncertainty = torch::zeros({num_gaussians}, 
                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
            
            // CG-SLAM Eq. 13 Approximation:
            // For each visible Gaussian, estimate its uncertainty contribution
            // based on rendered depth error in its pixel neighborhood
            
            // Get visibility info from rasterizer
            auto vis_mask = visibility_filter.to(torch::kBool);  // [N] - which Gaussians are visible
            auto visible_radii = radii.index({vis_mask});         // Radii of visible Gaussians
            
            // Compute uncertainty map: U = (d - D)² where d = rendered, D = GT depth
            auto depth_squeezed = depth.squeeze();  // [H, W]
            auto depth_error = depth_squeezed - gt_depth;
            auto uncertainty_map = depth_error * depth_error;
            auto gt_depth_valid = (gt_depth > RGBD_min_depth_) & (gt_depth < RGBD_max_depth_);
            
            // Mean uncertainty over valid pixels - use as baseline for all visible Gaussians
            auto valid_uncertainty = uncertainty_map.masked_select(gt_depth_valid);
            float mean_uncertainty = 0.0f;
            if (valid_uncertainty.numel() > 0) {
                mean_uncertainty = valid_uncertainty.mean().item<float>();
            }
            
            // Assign uncertainty to visible Gaussians based on their radius
            // Larger Gaussians (bigger radii) affect more pixels → inherit more uncertainty
            if (mean_uncertainty > 0.0f) {
                // Normalize radii to [0, 1] range
                auto visible_radii_float = visible_radii.to(torch::kFloat32);
                float max_radius = visible_radii_float.max().item<float>() + 1e-6f;
                auto normalized_radii = visible_radii_float / max_radius;
                
                // Per-Gaussian uncertainty = mean_uncertainty * (1 + normalized_radius)
                // This gives larger uncertainty to bigger Gaussians (more influential)
                auto visible_uncertainty = mean_uncertainty * (1.0f + normalized_radii);
                
                // Assign to full tensor using visibility mask
                per_gaussian_uncertainty.index_put_({vis_mask}, visible_uncertainty);
            }
            
            // Update Gaussian uncertainties with EMA (smooth across frames)
            gaussians_->updateUncertainty(per_gaussian_uncertainty, getIteration());
            
            // Count how many Gaussians will have opacity reduced
            auto& unc = gaussians_->getUncertainty();
            int num_high_before = (unc.depth_uncertainty > uncertainty_tau_).sum().item<int>();
            
            // CG-SLAM: Reduce opacity for high-uncertainty Gaussians (τ > threshold)
            // Paper: "primitives with νᵢ > τ will be manually reduced to a low-opacity level"
            gaussians_->uncertaintyPrune(uncertainty_tau_);
            
            // Track cumulative statistics (update static counter)
            static int64_t total_opacity_reduced = 0;
            static int prune_call_count = 0;
            int num_affected_this_call = std::min(num_high_before, static_cast<int>(num_gaussians * 0.05f));
            total_opacity_reduced += num_affected_this_call;
            prune_call_count++;
            
            // Log statistics every 500 iterations or when there are high-uncertainty Gaussians
            if (getIteration() % 500 == 0 || num_high_before > 0) {
                float unc_mean = unc.depth_uncertainty.mean().item<float>();
                float unc_max = unc.depth_uncertainty.max().item<float>();
                int num_high = (unc.depth_uncertainty > uncertainty_tau_).sum().item<int>();
                float pct_high = 100.0f * num_high / num_gaussians;
                float pct_cumulative = 100.0f * total_opacity_reduced / num_gaussians;
                
                std::cout << "[CG-SLAM Uncertainty] iter=" << getIteration()
                          << " | Gaussians=" << num_gaussians
                          << " | ν: mean=" << std::fixed << std::setprecision(4) << unc_mean 
                          << " max=" << unc_max
                          << " | high(>τ)=" << num_high << " (" << std::setprecision(1) << pct_high << "%)"
                          << " | τ=" << uncertainty_tau_ << std::endl;
                
                if (num_affected_this_call > 0) {
                    std::cout << "[CG-SLAM Uncertainty] Opacity reduced this iteration: " 
                              << num_affected_this_call 
                              << " | Cumulative total: " << total_opacity_reduced 
                              << " (" << std::setprecision(1) << pct_cumulative << "% of current Gaussians)"
                              << std::endl;
                }
            }
        }
        
        // ========================================================================
        // Fisher Information-based Uncertainty Quantification
        // Computes per-Gaussian importance based on gradient magnitude from backward pass
        // High Fisher score = Gaussian is important for reconstruction
        // Low Fisher score = Gaussian contributes little → candidate for pruning
        // ========================================================================
        if (fisher_config_.enabled) {
            int64_t num_gaussians = gaussians_->xyz_.size(0);
            
            // Compute Fisher scores at specified interval
            if (getIteration() % fisher_config_.update_interval == 0) {
                // Compute Fisher score from gradients
                // F_i ≈ ||∂L/∂xyz_i||² + λ_s||∂L/∂scale_i||² + ...
                auto fisher_score = fisher_info::computeFisherScore(
                    gaussians_->xyz_,
                    gaussians_->scaling_,
                    gaussians_->rotation_,
                    gaussians_->opacity_,
                    visibility_filter,
                    fisher_config_
                );
                
                // Update accumulated Fisher scores with EMA
                auto& uncertainty = gaussians_->getUncertainty();
                if (uncertainty.fisher_info_score.size(0) != num_gaussians) {
                    // Size mismatch - reinitialize
                    uncertainty.fisher_info_score = torch::zeros({num_gaussians},
                        torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
                }
                
                uncertainty.fisher_info_score = fisher_info::updateFisherEMA(
                    uncertainty.fisher_info_score,
                    fisher_score,
                    fisher_config_.ema_alpha
                );
                
                // Log Fisher statistics periodically
                if (getIteration() % 500 == 0) {
                    float fisher_mean = uncertainty.fisher_info_score.mean().item<float>();
                    float fisher_max = uncertainty.fisher_info_score.max().item<float>();
                    float fisher_min = uncertainty.fisher_info_score.min().item<float>();
                    int num_low = (uncertainty.fisher_info_score < fisher_config_.prune_threshold).sum().item<int>();
                    float pct_low = 100.0f * num_low / num_gaussians;
                    
                    std::cout << "[Fisher Info] iter=" << getIteration()
                              << " | Gaussians=" << num_gaussians
                              << " | F: mean=" << std::fixed << std::setprecision(4) << fisher_mean
                              << " min=" << fisher_min
                              << " max=" << fisher_max
                              << " | low(<τ)=" << num_low << " (" << std::setprecision(1) << pct_low << "%)"
                              << " | τ=" << fisher_config_.prune_threshold << std::endl;
                }
            }
            
            // Apply Fisher-based pruning at specified interval
            // Only prune after densification phase (when Gaussians are stable)
            if (getIteration() > opt_params_.densify_until_iter_ &&
                getIteration() % fisher_config_.prune_interval == 0) {
                
                auto& uncertainty = gaussians_->getUncertainty();
                
                // Generate prune mask based on Fisher Information
                auto prune_mask = fisher_info::generateFisherPruneMask(
                    uncertainty.fisher_info_score,
                    uncertainty.observation_count.to(torch::kFloat32),
                    fisher_config_
                );
                
                int num_to_prune = prune_mask.sum().item<int>();
                
                if (num_to_prune > 0) {
                    // Limit pruning to 5% per iteration to avoid sudden changes
                    int max_prune = static_cast<int>(num_gaussians * 0.05f);
                    
                    if (num_to_prune > max_prune) {
                        // Sort by Fisher score and only prune the lowest ones
                        auto [sorted_fisher, indices] = torch::sort(uncertainty.fisher_info_score);
                        auto prune_indices = indices.slice(0, 0, max_prune);
                        prune_mask = torch::zeros_like(prune_mask);
                        prune_mask.index_put_({prune_indices}, true);
                        num_to_prune = max_prune;
                    }
                    
                    // Prune low-information Gaussians
                    gaussians_->prunePoints(prune_mask);
                    uncertainty.prune(prune_mask);
                    
                    std::cout << "[Fisher Prune] iter=" << getIteration()
                              << " | Pruned " << num_to_prune << " low-information Gaussians"
                              << " | Remaining=" << gaussians_->xyz_.size(0) << std::endl;
                }
            }
        }
        // ========================================================================
        // END Fisher Information-based Uncertainty
        // ========================================================================
        
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

void GaussianMapper::setGTPoses(const std::vector<Sophus::SE3f>& gt_poses)
{
    gt_frame_poses_ = gt_poses;
    std::cout << "[GaussianMapper] GT poses set: " << gt_frame_poses_.size()
              << " frame poses available." << std::endl;
}

Sophus::SE3f GaussianMapper::getGTPoseForKF(unsigned long kf_id, const Sophus::SE3f& fallback)
{
    if (gt_frame_poses_.empty()) return fallback;
    if (!gt_alignment_computed_) return fallback;  // Alignment not ready yet

    // Look up the KeyFrame in ORB-SLAM3 atlas to get its frame index
    auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
    if (!pMap) return fallback;

    auto vpKFs = pMap->GetAllKeyFrames();
    for (auto* pKF : vpKFs) {
        if (pKF && pKF->mnId == kf_id) {
            unsigned long frameId = pKF->mnFrameId;
            if (frameId < gt_frame_poses_.size()) {
                // Apply alignment: convert GT pose from GT world to ORB-SLAM3 world
                // Tcw_aligned = Tcw_gt * S_inv  (where S maps GT_world -> ORB_world)
                return gt_frame_poses_[frameId] * gt_alignment_S_inv_;
            }
            break;
        }
    }
    return fallback;
}

void GaussianMapper::computeGTAlignment(const Sophus::SE3f& orb_Tcw, unsigned long frame_id)
{
    if (gt_alignment_computed_) return;
    if (frame_id >= gt_frame_poses_.size()) return;

    // Compute alignment transform S such that:
    //   p_orb_world = S * p_gt_world
    // From the first keyframe:
    //   Twc_orb = orb_Tcw^-1  (camera position in ORB world)
    //   Twc_gt  = gt_Tcw^-1   (camera position in GT world)
    //   S = Twc_orb * Twc_gt^-1
    // Then for any GT pose: Tcw_aligned = Tcw_gt * S^-1
    //   S^-1 = Twc_gt * Twc_orb^-1 = Twc_gt * orb_Tcw

    Sophus::SE3f gt_Tcw = gt_frame_poses_[frame_id];
    Sophus::SE3f Twc_gt = gt_Tcw.inverse();

    // S_inv = Twc_gt * Tcw_orb  (maps: ORB_world -> GT_world -> back)
    // We need: Tcw_aligned = Tcw_gt * S_inv
    // Where S_inv transforms from ORB_world -> GT_world
    // S = Twc_orb * Tcw_gt  (maps: GT_world -> camera -> ORB_world)
    // S_inv = Twc_gt * Tcw_orb  (maps: ORB_world -> camera -> GT_world)
    // Tcw_aligned = Tcw_gt_i * S_inv = Tcw_gt_i * Twc_gt_0 * Tcw_orb_0
    gt_alignment_S_inv_ = Twc_gt * orb_Tcw;
    gt_alignment_computed_ = true;

    std::cout << "[GaussianMapper] GT alignment computed from frame " << frame_id
              << ". ORB-SLAM3 pose: t=" << orb_Tcw.inverse().translation().transpose()
              << ", GT pose: t=" << Twc_gt.translation().transpose() << std::endl;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

void GaussianMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cout << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    handleNewKeyframe(kf);
                }
            }

            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
            // Mark the transformed points to avoid transforming more than once
            torch::Tensor point_not_transformed_flags =
                torch::full(
                    {gaussians_->xyz_.size(0)},
                    true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            int num_transformed = 0;
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                // In case new points are added in handleNewKeyframe()
                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        /*dim=*/0);
                // If kf is already in the scene, evaluate the change in pose,
                // if too large we perform loop correction on its visible model points.
                // If not in the scene, create a new one.
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    // If is loop closure kf
// if (std::get<4>(kf)) {
// renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                        Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                        Sophus::SE3f inv_pose = pose.inverse();
                        Sophus::SE3f diff_pose = inv_pose * original_pose;
                        bool large_rot = !diff_pose.rotationMatrix().isApprox(
                            Eigen::Matrix3f::Identity(), large_rot_th_);
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                            1.0, large_trans_th_);
                        if (large_rot || large_trans) {
                            std::cout << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
                                    << kfid << std::endl;
                            diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                            diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                            diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                            {
                                std::unique_lock<std::mutex> lock_render(mutex_render_);
                                gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                    point_not_transformed_flags,
                                    diff_pose_tensor,
                                    pkf->world_view_transform_,
                                    pkf->full_proj_transform_,
                                    pkf->creation_iter_,
                                    stableNumIterExistence(),
                                    num_transformed,
                                    loop_kf_scale); // selected xyz *= s
                            }
                            // Give loop keyframes times of use
                            increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
// renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
// std::cout<<num_transformed<<std::endl;
                        }
// }
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();
// if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                }
                else {
                    handleNewKeyframe(kf);
                }
            }
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
// keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));

            // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }

            // Mark this iteration
            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      << std::endl;

            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            if (initial_mapped_) {
                // Apply the scaled transformation on gaussian model points
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                // Apply the scaled transformation to the scene
                scene_->applyScaledTransformation(s, T);
            }
            else { // TODO: the workflow should not come here, delete this branch
                // Apply the scaled transformation to the cached points
                for (auto& pt : scene_->cached_point_cloud_) {
                    // pt <- (s * Ryw * pt + tyw)
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }

                // Apply the scaled transformation on gaussian keyframes
                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}

void GaussianMapper::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // ====================================================================
    // Molding-GS: Fuse depth into Local TSDF (ASYNC — non-blocking)
    // integrateDepthFrame is CPU-heavy. We launch it in a background future
    // so the main mapping loop is not blocked and can keep training/densifying.
    // ====================================================================
    if (tsdf_config_.enabled && local_tsdf_ && sensor_type_ == RGBD &&
        imgAux_undistorted.data) {
        // Check if previous fusion is done — never block the GPU loop.
        bool prev_done = !tsdf_fusion_future_.valid() ||
                         tsdf_fusion_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        if (prev_done) {
            // Safe to launch a new async fusion for this keyframe
            cv::Mat depth_copy = imgAux_undistorted.clone();
            Camera cam_tsdf = scene_->cameras_.at(std::get<1>(kf));
            float fx_tsdf = cam_tsdf.params_[0];
            float fy_tsdf = cam_tsdf.params_[1];
            float cx_tsdf = cam_tsdf.params_[2];
            float cy_tsdf = cam_tsdf.params_[3];
            Sophus::SE3f Tcw = std::get<2>(kf);
            Eigen::Matrix4f Twc_eigen = Tcw.inverse().matrix();
            Eigen::Vector3f cam_pos = Twc_eigen.block<3,1>(0, 3);
            float min_depth = RGBD_min_depth_;
            float max_depth = RGBD_max_depth_;
            float max_radius = tsdf_config_.max_radius;
            local_tsdf::LocalTSDF* tsdf_ptr = local_tsdf_.get();

            // Launch async fusion — does NOT block the mapping thread
            tsdf_fusion_future_ = std::async(std::launch::async, [=]() {
                try {
                    tsdf_ptr->integrateDepthFrame(
                        depth_copy, Twc_eigen,
                        fx_tsdf, fy_tsdf, cx_tsdf, cy_tsdf,
                        min_depth, max_depth);
                    tsdf_ptr->evictDistant(cam_pos, max_radius);
                } catch (const std::exception& e) {
                    std::cerr << "[Molding-GS] TSDF fusion error: " << e.what() << std::endl;
                }
            });

            static int tsdf_fusion_count = 0;
            tsdf_fusion_count++;
            if (tsdf_fusion_count % 10 == 0 || tsdf_fusion_count <= 3) {
                std::cout << "[Molding-GS] TSDF fusion #" << tsdf_fusion_count
                          << " (async) | voxels≈" << local_tsdf_->numVoxels()
                          << " | mem≈" << (local_tsdf_->memoryUsageBytes() / 1024)
                          << "KB" << std::endl;
            }
        }
        // If prev_done == false: skip this keyframe's fusion to keep GPU running
    }

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}

void GaussianMapper::generateKfidRandomShuffle()
{
// if (viewpoint_sliding_window_.empty())
//     return;

// std::size_t sliding_window_size = viewpoint_sliding_window_.size();
// kfid_shuffle_.resize(sliding_window_size);
// std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
// std::mt19937 g(rd_());
// std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

// auto t2 = std::chrono::steady_clock::now();
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
// std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
{
    std::unordered_set<unsigned long> kfids =
        pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    std::vector<unsigned long> kfids_to_erase;
    std::size_t nkfs = scene_->keyframes().size();
    kfids_to_erase.reserve(nkfs);
    for (auto& kfit : scene_->keyframes()) {
        unsigned long kfid = kfit.first;
        if (kfids.find(kfid) == kfids.end()) {
            kfids_to_erase.emplace_back(kfid);
        }
    }

    for (auto& kfid : kfids_to_erase) {
        scene_->keyframes().erase(kfid);
    }
}

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREO:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBD:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // Determine point valid flags: DepthBackproject OR Guided Filter dense OR ORB-sparse-only
        torch::Tensor point_valid_flags;

        if (depth_backproject_config_.enabled) {
            if (depth_backproject_config_.mode == 1) {
                // Adaptive stride: edge-aware block-based sampling
                point_valid_flags = depth_backproject::selectAdaptiveStridePixels(
                    depth, img_rgb_gpu,
                    pkf->image_width_, pkf->image_height_,
                    depth_backproject_config_, device_type_);
            } else {
                // Uniform stride
                point_valid_flags = depth_backproject::selectStrideDepthPixels(
                    depth, pkf->image_width_, pkf->image_height_,
                    depth_backproject_config_, device_type_);
                std::cout << "[DepthBackproject Uniform] KF " << pkf->fid_
                          << " | stride=" << depth_backproject_config_.stride
                          << " | selected=" << point_valid_flags.sum().item<int>()
                          << " pts" << std::endl;
            }
        } else if (guided_depth_config_.enabled) {
            // Edge-aware dense sampling from sensor depth
            point_valid_flags = guided_depth::selectDenseDepthPixels(
                img_rgb_gpu, img_depth_gpu,
                pkf->kps_pixel_,
                pkf->image_width_, pkf->image_height_,
                RGBD_min_depth_, RGBD_max_depth_,
                guided_depth_config_, device_type_);

            if (!point_valid_flags.defined() || point_valid_flags.numel() == 0) {
                // Fallback: use ORB-only sparse flags
                std::cout << "[Guided Depth] Fallback to ORB-sparse for KF " << pkf->fid_ << std::endl;
                point_valid_flags = torch::full(
                    {depth.size(0)}, false,
                    torch::TensorOptions().dtype(torch::kBool).device(device_type_));
                int nkps_twice = pkf->kps_pixel_.size();
                int width = pkf->image_width_;
                for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
                    int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                              static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
                    point_valid_flags[idx] = true;
                }
                point_valid_flags = torch::logical_and(
                    point_valid_flags,
                    torch::where(depth > RGBD_min_depth_, true, false));
                point_valid_flags = torch::logical_and(
                    point_valid_flags,
                    torch::where(depth < RGBD_max_depth_, true, false));
            }
            // No need for extra depth range check — selectDenseDepthPixels already checks
        } else {
            // Original ORB-only sparse behavior
            point_valid_flags = torch::full(
                {depth.size(0)}, false,
                torch::TensorOptions().dtype(torch::kBool).device(device_type_));
            int nkps_twice = pkf->kps_pixel_.size();
            int width = pkf->image_width_;
            for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
                int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                          static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
                point_valid_flags[idx] = true;
            }
            point_valid_flags = torch::logical_and(
                point_valid_flags,
                torch::where(depth > RGBD_min_depth_, true, false));
            point_valid_flags = torch::logical_and(
                point_valid_flags,
                torch::where(depth < RGBD_max_depth_, true, false));
        }

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to get 3D points
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Wavelet-guided Gaussian Initialization: Add extra points at edge regions
        if (wavelet_config_.init_enabled) {
            try {
                // Get image dimensions
                int H = pkf->image_height_;
                int W = pkf->image_width_;
                
                // Convert RGB to grayscale tensor for wavelet decomposition
                // rgb is [H*W, 3], reshape to [3, H, W] for wavelet
                auto rgb_reshaped = rgb.view({H, W, 3}).permute({2, 0, 1}).contiguous();
                
                // Compute wavelet decomposition
                auto wavelet_decomp = wavelet::haarDecompose2D(rgb_reshaped);
                
                // Compute edge strength: |LH| + |HL| + |HH|
                auto edge_map = torch::abs(wavelet_decomp.LH).mean(0) + 
                                torch::abs(wavelet_decomp.HL).mean(0) + 
                                torch::abs(wavelet_decomp.HH).mean(0);
                
                // Upsample edge_map back to original resolution (wavelet output is H/2 x W/2)
                auto edge_map_upsampled = torch::nn::functional::interpolate(
                    edge_map.unsqueeze(0).unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{H, W})
                        .mode(torch::kBilinear)
                        .align_corners(false)
                ).squeeze();  // [H, W]
                
                // Flatten and apply validity mask
                auto edge_flat = edge_map_upsampled.flatten().index({point_valid_flags});
                
                // Find high-edge pixels above threshold
                auto edge_mask = edge_flat > wavelet_config_.init_edge_threshold;
                int num_edge_points = edge_mask.sum().item<int>();
                
                if (num_edge_points > 0) {
                    // Get edge points and colors
                    auto edge_points = points3D_valid.index({edge_mask});
                    auto edge_colors = colors_valid.index({edge_mask});
                    
                    // Limit extra points
                    int max_extra = wavelet_config_.init_max_extra_points;
                    if (num_edge_points > max_extra) {
                        // Random sampling
                        auto indices = torch::randperm(num_edge_points, 
                            torch::TensorOptions().device(device_type_)).slice(0, 0, max_extra);
                        edge_points = edge_points.index({indices});
                        edge_colors = edge_colors.index({indices});
                        num_edge_points = max_extra;
                    }
                    
                    // Add small random offset to avoid exact duplicates
                    auto offset = torch::randn_like(edge_points) * 0.002f;
                    edge_points = edge_points + offset;
                    
                    // Merge with original points
                    points3D_valid = torch::cat({points3D_valid, edge_points}, 0);
                    colors_valid = torch::cat({colors_valid, edge_colors}, 0);
                    
                    std::cout << "[Wavelet Init] KF " << pkf->fid_ 
                              << " | Added " << num_edge_points 
                              << " extra edge points (total: " << points3D_valid.size(0) << ")" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Wavelet Init] Warning: " << e.what() << std::endl;
            }
        }

        // Geometry-Aware Gaussian Initialization: compute better init params
        torch::Tensor geo_rotations, geo_scale_mods, geo_opacities;
        bool has_geo_aware = false;
        if (geo_aware_config_.enabled) {
            try {
                // Download depth map to CPU for normal computation
                cv::Mat depth_cpu;
                img_depth_gpu.download(depth_cpu);
                cv::Mat depth_float;
                if (depth_cpu.channels() > 1) {
                    std::vector<cv::Mat> channels;
                    cv::split(depth_cpu, channels);
                    channels[0].convertTo(depth_float, CV_32FC1);
                } else {
                    depth_cpu.convertTo(depth_float, CV_32FC1);
                }

                cv::Mat rgb_cpu;
                img_rgb_gpu.download(rgb_cpu);

                // Get camera intrinsics
                Camera& cam = scene_->cameras_.at(pkf->camera_id_);
                float fx = cam.params_[0];
                float fy = cam.params_[1];
                float cx = cam.params_[2];
                float cy = cam.params_[3];

                has_geo_aware = geo_aware::computeGeoAwareParams(
                    depth_float, rgb_cpu, point_valid_flags,
                    points3D_valid, fx, fy, cx, cy,
                    pkf->image_width_, pkf->image_height_,
                    geo_aware_config_, device_type_,
                    geo_rotations, geo_scale_mods, geo_opacities);

                // Handle wavelet extra points: append default params for them
                if (has_geo_aware && points3D_valid.size(0) > geo_rotations.size(0)) {
                    int n_extra = points3D_valid.size(0) - geo_rotations.size(0);
                    // Default identity rotation for extra points
                    auto extra_rots = torch::zeros({n_extra, 4},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
                    extra_rots.index({torch::indexing::Slice(), 0}) = 1.0f;
                    geo_rotations = torch::cat({geo_rotations, extra_rots}, 0);
                    // Default no-flatten for extra points
                    auto extra_scales = torch::zeros({n_extra, 3},
                        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
                    geo_scale_mods = torch::cat({geo_scale_mods, extra_scales}, 0);
                    // Default opacity for extra points
                    float default_opa = std::log(0.1f / (1.0f - 0.1f));
                    auto extra_opas = torch::full({n_extra, 1}, default_opa,
                        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
                    geo_opacities = torch::cat({geo_opacities, extra_opas}, 0);
                }
            } catch (const std::exception& e) {
                std::cerr << "[GeoAware] Warning: " << e.what() << std::endl;
                has_geo_aware = false;
            }
        }
        // [ConeGS] Old EGD block removed — densification is now handled
        // by ConeGS error-guided insertion in trainForOneIteration().

        // Add new points to the cache
        if (points3D_valid.size(0) > 0) {
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
            if (has_geo_aware) {
                depth_cache_rotations_ = geo_rotations;
                depth_cache_scale_mods_ = geo_scale_mods;
                depth_cache_opacities_ = geo_opacities;
            }
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
            if (has_geo_aware) {
                depth_cache_rotations_ = torch::cat({depth_cache_rotations_, geo_rotations}, /*dim=*/0);
                depth_cache_scale_mods_ = torch::cat({depth_cache_scale_mods_, geo_scale_mods}, /*dim=*/0);
                depth_cache_opacities_ = torch::cat({depth_cache_opacities_, geo_opacities}, /*dim=*/0);
            }
        }
        ++depth_cached_;   // Only increment when we actually cached points
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        // Add new points to the model — only if cache has data
        if (depth_cache_points_.defined() && depth_cache_points_.size(0) > 0) {
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            if (geo_aware_config_.enabled && depth_cache_rotations_.defined() &&
                depth_cache_rotations_.size(0) == depth_cache_points_.size(0)) {
                gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_,
                                       depth_cache_rotations_, depth_cache_scale_mods_,
                                       depth_cache_opacities_, getIteration());
            } else {
                gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
            }
        }
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        torch::Tensor &rendered_depth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::filesystem::path result_depth_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }

    // Save rendered depth maps
    if (record_rendered_image_ && rendered_depth.defined() && rendered_depth.numel() > 0) {
        // Convert depth tensor to cv::Mat - squeeze to remove channel dimension
        torch::Tensor depth_squeezed = rendered_depth.squeeze().cpu();
        auto depth_cv = tensor_utils::torchTensor2CvMat_Float32(depth_squeezed);
        
        // Save as 16-bit PNG with configurable scale (TUM=5000, Replica=6553.5)
        cv::Mat depth_16u;
        depth_cv.convertTo(depth_16u, CV_16U, depth_save_scale_);
        cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_depth.png"), depth_16u);
        
        // Save visualization with colormap
        cv::Mat depth_vis;
        cv::normalize(depth_cv, depth_vis, 0, 255, cv::NORM_MINMAX);
        depth_vis.convertTo(depth_vis, CV_8U);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
        cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_depth_vis.jpg"), depth_vis);
    }
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    // MIG/ESC: Updated to match 11-element return type (includes T_map, depths, cov2D)
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor, at::Tensor, at::Tensor, at::Tensor, 
               at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::filesystem::path result_depth_dir,
    std::string name_suffix)
{
    auto start_timing = std::chrono::steady_clock::now();
    auto render_pkg = GaussianRenderer::render(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    auto rendered_depth = std::get<4>(render_pkg);  // Extract depth (index 4)
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, rendered_depth, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, result_depth_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path depth_dir = result_dir / "depth";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, depth_dir, "");

        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapper::renderAndRecordAllTrajectoryPoses(
    std::filesystem::path trajectory_file,
    std::string name_suffix)
{
    // Read TUM trajectory file
    std::ifstream traj_file(trajectory_file);
    if (!traj_file.is_open()) {
        std::cerr << "[GaussianMapper] Cannot open trajectory file: " << trajectory_file << std::endl;
        return;
    }

    std::vector<std::tuple<unsigned long, Eigen::Quaterniond, Eigen::Vector3d>> poses;
    std::string line;
    while (std::getline(traj_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double timestamp, tx, ty, tz, qx, qy, qz, qw;
        if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
        unsigned long fid = static_cast<unsigned long>(timestamp);
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);
        poses.emplace_back(fid, q, t);
    }
    traj_file.close();

    if (poses.empty()) {
        std::cerr << "[GaussianMapper] No poses found in trajectory file" << std::endl;
        return;
    }

    std::cout << "[GaussianMapper] Rendering depth for " << poses.size() << " trajectory poses..." << std::endl;

    // Create output directories
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path depth_dir = result_dir / "all_depths";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir)

    // Get camera from first keyframe
    if (scene_->keyframes().empty()) {
        std::cerr << "[GaussianMapper] No keyframes available for camera params" << std::endl;
        return;
    }
    auto first_kf = scene_->keyframes().begin()->second;

    // Render depth for each pose
    int progress = 0;
    for (const auto& [fid, q, t] : poses) {
        // Create temporary keyframe-like object for rendering
        auto temp_kf = std::make_shared<GaussianKeyframe>();
        temp_kf->fid_ = fid;
        temp_kf->zfar_ = z_far_;
        temp_kf->znear_ = z_near_;
        temp_kf->setPose(q, t);
        temp_kf->setCameraParams(scene_->cameras_.at(first_kf->camera_id_));
        temp_kf->computeTransformTensors();

        // Render
        auto render_pkg = GaussianRenderer::render(
            temp_kf,
            temp_kf->image_height_,
            temp_kf->image_width_,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
        auto rendered_depth = std::get<4>(render_pkg);

        // Save depth
        if (rendered_depth.defined() && rendered_depth.numel() > 0) {
            torch::Tensor depth_squeezed = rendered_depth.squeeze().cpu();
            auto depth_cv = tensor_utils::torchTensor2CvMat_Float32(depth_squeezed);
            cv::Mat depth_16u;
            depth_cv.convertTo(depth_16u, CV_16U, depth_save_scale_);
            std::string filename = std::to_string(fid) + "_depth.png";
            cv::imwrite(depth_dir / filename, depth_16u);
        }

        progress++;
        if (progress % 100 == 0) {
            std::cout << "[GaussianMapper] Rendered " << progress << "/" << poses.size() << " frames" << std::endl;
        }
    }

    std::cout << "[GaussianMapper] Finished rendering all " << poses.size() << " trajectory poses" << std::endl;
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}
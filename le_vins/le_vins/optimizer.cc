/*
 * Copyright (C) 2024 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "le_vins/optimizer.h"

#include "common/logging.h"
#include "common/misc.h"
#include "common/timecost.h"

#include "visual/tracking.h"

#include "factors/depth_factor.h"
#include "factors/marginalization_factor.h"
#include "factors/marginalization_info.h"
#include "factors/pose_manifold.h"
#include "factors/reprojection_factor.h"
#include "factors/residual_block_info.h"
#include "factors/time_delay_factor.h"

#include "preintegration/imu_error_factor.h"
#include "preintegration/imu_mix_prior_factor.h"
#include "preintegration/imu_pose_prior_factor.h"
#include "preintegration/imu_zero_velocity_factor.h"
#include "preintegration/preintegration.h"
#include "preintegration/preintegration_factor.h"

#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

Optimizer::Optimizer(const string &configfile, std::shared_ptr<IntegrationParameters> parameters,
                     PreintegrationOptions options, visual::VisualMap::Ptr visual_map, visual::Camera::Ptr camera)
    : integration_parameters_(std::move(parameters))
    , preintegration_options_(options)
    , visual_map_(std::move(visual_map))
    , camera_(std::move(camera)) {

    // 配置参数
    YAML::Node config;
    vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    // 优化参数
    optimize_reprojection_std_  = config["optimizer"]["optimize_reprojection_std"].as<double>();
    optimize_windows_size_      = config["optimizer"]["optimize_window_size"].as<size_t>();
    optimize_half_windows_size_ = optimize_windows_size_ / 2;

    optimize_estimate_cam_extrinsic_ = config["optimizer"]["optimize_estimate_cam_extrinsic"].as<bool>();
    optimize_estimate_cam_td_        = config["optimizer"]["optimize_estimate_cam_td"].as<bool>();
    optimize_cam_extrinsic_accurate_ = config["optimizer"]["optimize_cam_extrinsic_accurate"].as<bool>();

    // 归一化相机坐标系下
    optimize_reprojection_error_std_ = optimize_reprojection_std_ / camera_->focalLength();

    gb_prior_std_ = config["imu"]["gb_prior_std"].as<double>();
    ab_prior_std_ = config["imu"]["ab_prior_std"].as<double>();
    imudatarate_  = config["imu"]["imudatarate"].as<double>();

    // 相机参数
    is_use_lidar_depth_ = config["visual"]["is_use_lidar_depth"].as<bool>();
    vecdata             = config["visual"]["q_b_c"].as<vector<double>>();
    Quaterniond q_b_c   = Eigen::Quaterniond(vecdata.data()).normalized();
    vecdata             = config["visual"]["t_b_c"].as<vector<double>>();
    Vector3d t_b_c      = Eigen::Vector3d(vecdata.data());
    double td_b_c       = config["visual"]["td_b_c"].as<double>();

    memcpy(extrinsic_b_c_, t_b_c.data(), sizeof(double) * 3);
    memcpy(extrinsic_b_c_ + 3, q_b_c.coeffs().data(), sizeof(double) * 4); // x, y, z, w
    extrinsic_b_c_[7] = td_b_c;

    // 统计参数
    iterations_.resize(2);
    timecosts_.resize(4);
    outliers_.resize(3);

    LOGI << "Optimizer is constructed";
}

bool Optimizer::optimization(const deque<double> &timelist, deque<StateData> &statedatalist,
                             deque<PreBasePtr> &preintegrationlist) {
    static int first_num_iterations  = 5;
    static int second_num_iterations = 10;

    TimeCost timecost;

    // 方便后续移除粗差因子
    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type         = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 8;

    // 状态参数
    addStateParameters(problem, statedatalist);

    // 重投影参数
    addVisualParameters(problem);
    LOGI << "Add " << invdepthlist_.size() << " visual landmarks to optimizer";

    // 边缘化残差
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // 预积分残差
    addImuFactors(problem, statedatalist, preintegrationlist);

    // 视觉重投影残差
    Residuals visual_residuals;
    visual_residuals = addReprojectionFactors(problem, true, timelist, statedatalist);

    string msg = absl::StrFormat("Add %llu preintegration", preintegrationlist.size());
    absl::StrAppendFormat(&msg, ", %llu visual", visual_residuals.size());
    LOGI << msg << " factors";

    // 第一次优化
    {
        timecost.restart();

        // 求解最小二乘
        solver.Solve(options, &problem, &summary);
        // LOGI << summary.BriefReport();

        iterations_[0] = summary.num_successful_steps;
        timecosts_[0]  = timecost.costInMillisecond();
    }

    // 粗差检测和剔除
    {
        // Detect outlier visual factors
        ResidualOutliers visual_outliers;
        visual_outliers = visualOutlierCullingByChi2(problem, visual_residuals);

        // Remove factors in the final

        int num_outliers = 0;
        // Remove outliers only
        for (const auto &outlier : visual_outliers) {
            if (outlier.second) {
                problem.RemoveResidualBlock(outlier.first);
                num_outliers++;
            }
        }
        LOGI << "Remove " << num_outliers << " reprojection outlier factors";
    }

    // 第二次优化
    {
        // 第二次迭代次数
        options.max_num_iterations = second_num_iterations;

        timecost.restart();

        // 求解最小二乘
        solver.Solve(options, &problem, &summary);
        // LOGI << summary.BriefReport();

        iterations_[1] = summary.num_successful_steps;
        timecosts_[1]  = timecost.costInMillisecond();

        if (!visual_map_->isWindowFull()) {
            // 进行必要的重积分
            doReintegration(preintegrationlist, statedatalist);
        }
    }

    // 更新参数, 必须的
    updateParametersFromOptimizer(timelist, statedatalist);

    // 移除粗差路标点
    visualLandmarkOutlierCulling();

    LOGI << "Optimization costs " << timecosts_[0] << " ms and " << timecosts_[1] << ", with iteration "
         << iterations_[0] << " and " << iterations_[1];
    return true;
}

void Optimizer::updateParametersFromOptimizer(const deque<double> &timelist, const deque<StateData> &statedatalist) {
    // 视觉处理参数
    if (!(visual_map_->keyframes().size() < 2)) {
        // 先更新外参, 更新位姿需要外参
        Pose pose_b_c;
        pose_b_c.t[0] = extrinsic_b_c_[0];
        pose_b_c.t[1] = extrinsic_b_c_[1];
        pose_b_c.t[2] = extrinsic_b_c_[2];

        Quaterniond q_b_c = Quaterniond(extrinsic_b_c_[6], extrinsic_b_c_[3], extrinsic_b_c_[4], extrinsic_b_c_[5]);
        pose_b_c.R        = Rotation::quaternion2matrix(q_b_c.normalized());

        // 保证系统稳定
        extrinsic_b_c_[7] = visual_map_->latestKeyFrame()->timeDelayEst();
        if (optimize_estimate_cam_td_ && (fabs(extrinsic_b_c_[7]) > 0.2)) {
            LOGW << "Estimate large td_b_c " << Logging::doubleData(extrinsic_b_c_[7]) << ", and td_b_c is reset";
            extrinsic_b_c_[7] = 0;
        }

        // 更新关键帧的位姿
        for (auto &keyframe : visual_map_->keyframes()) {
            auto &frame = keyframe.second;
            auto index  = getStateDataIndex(timelist, frame->stamp());
            if (index < 0) {
                continue;
            }

            IntegrationState state = Preintegration::stateFromData(statedatalist[index], preintegration_options_);
            frame->setPose(MISC::stateToPoseWithExtrinsic(state, pose_b_c));
        }

        // 更新路标点的深度和位置
        for (const auto &landmark : visual_map_->landmarks()) {
            const auto &mappoint = landmark.second;
            if (!mappoint || mappoint->isOutlier()) {
                continue;
            }

            auto frame = mappoint->referenceFrame();
            // 由于前端修改参考帧导致路标点暂时失效
            if (!frame || !visual_map_->isKeyFrameInMap(frame)) {
                continue;
            }

            // 未参与优化的无效路标点
            if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
                continue;
            }

            double invdepth = invdepthlist_[mappoint->id()];
            double depth    = 1.0 / invdepth;

            auto pc0      = camera_->pixel2cam(mappoint->referenceKeypoint());
            Vector3d pc00 = {pc0.x(), pc0.y(), 1.0};
            pc00 *= depth;

            mappoint->pos() = camera_->cam2world(pc00, mappoint->referenceFrame()->pose());
            mappoint->updateDepth(depth);
        }
    }
}

bool Optimizer::marginalization(deque<double> &timelist, deque<StateData> &statedatalist,
                                deque<PreBasePtr> &preintegrationlist) {
    TimeCost timecost;

    size_t num_marg = 0;
    vector<ulong> keyframeids;
    //  按时间先后排序的关键帧
    keyframeids          = visual_map_->orderedKeyFrames();
    auto latest_keyframe = visual_map_->latestKeyFrame();

    latest_keyframe->setKeyFrameState(visual::KEYFRAME_NORMAL);

    // 对齐到保留的最后一个关键帧, 可能移除多个预积分对象
    auto frame = visual_map_->keyframes().find(keyframeids[1])->second;
    num_marg   = static_cast<size_t>(getStateDataIndex(timelist, frame->stamp()));

    // 保留的时间节点
    double last_time = timelist[num_marg];

    // 边缘化信息
    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    std::unordered_map<long, long> parameters_ids;
    long parameters_id = 0;
    {
        // 边缘化参数
        for (auto &last_marginalization_parameter_block : last_marginalization_parameter_blocks_) {
            parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] = parameters_id++;
        }

        // 外参参数
        parameters_ids[reinterpret_cast<long>(extrinsic_b_c_)] = parameters_id++;
        for (size_t k = 0; k < keyframeids.size(); k++) {
            auto frame = visual_map_->keyframes().at(keyframeids[k]);
            parameters_ids[reinterpret_cast<long>(&frame->timeDelayEst())] = parameters_id++;
        }

        // 位姿参数
        for (const auto &statedata : statedatalist) {
            parameters_ids[reinterpret_cast<long>(statedata.pose)] = parameters_id++;
            parameters_ids[reinterpret_cast<long>(statedata.mix)]  = parameters_id++;
        }

        // 最老关键帧的逆深度参数
        auto frame           = visual_map_->keyframes().at(keyframeids[0]);
        const auto &features = frame->features();
        for (auto const &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
                continue;
            }

            // 参考帧非边缘化帧, 不边缘化
            if (mappoint->referenceFrame() != frame) {
                continue;
            }

            // 对应的逆深度数据地址
            double *invdepth = &invdepthlist_[mappoint->id()];

            parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        }

        // 更新参数块的特定ID, 必要的
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // 边缘化因子
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {

        vector<int> marginalized_index;
        for (size_t i = 0; i < num_marg; i++) {
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
                if (last_marginalization_parameter_blocks_[k] == statedatalist[i].pose ||
                    last_marginalization_parameter_blocks_[k] == statedatalist[i].mix) {
                    marginalized_index.push_back((int) k);
                }
            }
        }

        if (optimize_estimate_cam_td_) {
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
                if (last_marginalization_parameter_blocks_[k] ==
                    &visual_map_->keyframes().at(keyframeids[0])->timeDelayEst()) {
                    marginalized_index.push_back((int) k);
                    break;
                }
            }
        }

        auto factor   = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr, last_marginalization_parameter_blocks_,
                                                            marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 预积分因子
    for (size_t k = 0; k < num_marg; k++) {
        // 由于会移除多个预积分, 会导致出现保留和移除同时出现, 判断索引以区分
        vector<int> marg_index;
        if (k == (num_marg - 1)) {
            marg_index = {0, 1};
        } else {
            marg_index = {0, 1, 2, 3};
        }

        auto factor = std::make_shared<PreintegrationFactor>(preintegrationlist[k]);
        auto residual =
            std::make_shared<ResidualBlockInfo>(factor, nullptr,
                                                vector<double *>{statedatalist[k].pose, statedatalist[k].mix,
                                                                 statedatalist[k + 1].pose, statedatalist[k + 1].mix},
                                                marg_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 先验约束因子
    if (is_use_prior_) {
        auto pose_factor   = std::make_shared<ImuPosePriorFactor>(pose_prior_, pose_prior_std_);
        auto pose_residual = std::make_shared<ResidualBlockInfo>(
            pose_factor, nullptr, vector<double *>{statedatalist[0].pose}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(pose_residual);

        auto mix_factor   = std::make_shared<ImuMixPriorFactor>(preintegration_options_, mix_prior_, mix_prior_std_);
        auto mix_residual = std::make_shared<ResidualBlockInfo>(mix_factor, nullptr,
                                                                vector<double *>{statedatalist[0].mix}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(mix_residual);

        // 相机IMU外参先验
        if (optimize_estimate_cam_extrinsic_) {
            auto extrinsic_cam_factor =
                std::make_shared<ImuPosePriorFactor>(extrinsic_cam_prior_, extrinsic_cam_prior_std_);
            auto extrinsic_cam_residual = std::make_shared<ResidualBlockInfo>(
                extrinsic_cam_factor, nullptr, vector<double *>{extrinsic_b_c_}, vector<int>{});
            marginalization_info->addResidualBlockInfo(extrinsic_cam_residual);
        }
        is_use_prior_ = false;
    }

    {
        // 重投影因子, 最老的关键帧
        auto frame           = visual_map_->keyframes().at(keyframeids[0]);
        const auto &features = frame->features();

        // 对重投影误差添加loss函数
        auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
        for (auto const &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
                continue;
            }

            // 参考帧
            auto ref_frame = mappoint->referenceFrame();
            // 参考帧非边缘化帧, 不边缘化
            if (ref_frame != frame) {
                continue;
            }

            auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
            size_t ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
            if (ref_frame_index < 0) {
                continue;
            }

            // 逆深度
            double *invdepth = &invdepthlist_[mappoint->id()];

            // 激光深度辅助, 深度约束因子
            if (is_use_lidar_depth_) {
                if (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED) {
                    auto factor = std::make_shared<ceres::AutoDiffCostFunction<DepthFactor, 1, 1>>(
                        new DepthFactor(mappoint->lidarDepth(), visual::Tracking::ASSOCIATE_DEPTH_STD));
                    auto residual = std::make_shared<ResidualBlockInfo>(factor, loss_function,
                                                                        vector<double *>{invdepth}, vector<int>{0});
                    marginalization_info->addResidualBlockInfo(residual);
                }
            }

            auto ref_feature = ref_frame->features().find(mappoint->id())->second;

            // 所有的观测
            for (const auto &observation : mappoint->observations()) {
                auto obs_feature = observation.lock();
                // 无效特征
                if (!obs_feature || obs_feature->isOutlier()) {
                    continue;
                }
                auto obs_frame = obs_feature->getFrame();
                // 非关键帧特征, 参考帧
                if (!obs_frame || !obs_frame->isKeyFrame() || !visual_map_->isKeyFrameInMap(obs_frame) ||
                    (obs_frame == ref_frame)) {
                    continue;
                }

                // 观测帧
                auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
                size_t obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());

                if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                    LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                         << Logging::doubleData(obs_frame->stamp());
                    continue;
                }

                auto factor = std::make_shared<ReprojectionFactor>(
                    ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
                    ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_error_std_);
                auto residual = std::make_shared<ResidualBlockInfo>(
                    factor, nullptr,
                    vector<double *>{statedatalist[ref_frame_index].pose, statedatalist[obs_frame_index].pose,
                                     extrinsic_b_c_, invdepth, &ref_frame->timeDelayEst(), &obs_frame->timeDelayEst()},
                    vector<int>{0, 3, 4});
                marginalization_info->addResidualBlockInfo(residual);
            }
        }

        if (optimize_estimate_cam_td_) {
            auto frame_pre = visual_map_->keyframes().at(keyframeids[0]);
            auto frame_cur = visual_map_->keyframes().at(keyframeids[1]);
            auto factor    = std::make_shared<TimeDelayFactor>(frame_cur->stamp() - frame_pre->stamp());
            auto residual  = std::make_shared<ResidualBlockInfo>(
                factor, nullptr, vector<double *>{&frame_pre->timeDelayEst(), &frame_cur->timeDelayEst()},
                vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    // 边缘化处理
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID
    std::unordered_map<long, double *> address;
    for (size_t k = num_marg; k < statedatalist.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(statedatalist[k].pose)]] = statedatalist[k].pose;
        address[parameters_ids[reinterpret_cast<long>(statedatalist[k].mix)]]  = statedatalist[k].mix;
    }
    address[parameters_ids[reinterpret_cast<long>(extrinsic_b_c_)]] = extrinsic_b_c_;
    for (size_t k = 1; k < keyframeids.size(); k++) {
        auto frame = visual_map_->keyframes().at(keyframeids[k]);
        address[parameters_ids[reinterpret_cast<long>(&frame->timeDelayEst())]] = &frame->timeDelayEst();
    }

    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_             = std::move(marginalization_info);

    // 移除边缘化的数据

    {
        // 保存移除的路标点
        auto frame           = visual_map_->keyframes().at(keyframeids[0]);
        const auto &features = frame->features();

        fixed_visual_mappoints_.clear();
        for (const auto &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
                continue;
            }
            fixed_visual_mappoints_.push_back(mappoint->pos());
        }

        // 关键帧
        visual_map_->removeKeyFrame(frame, true);
    }

    // 预积分观测及时间状态, 放最后, 需要关键帧索引需要timelist
    for (size_t k = 0; k < num_marg; k++) {
        timelist.pop_front();
        statedatalist.pop_front();
        preintegrationlist.pop_front();
    }

    timecosts_[2] = timecost.costInMillisecond();

    LOGI << "Marginalize " << num_marg << " states, last time " << Logging::doubleData(last_time) << ", costs "
         << timecosts_[2];

    return true;
}

void Optimizer::addStateParameters(ceres::Problem &problem, deque<StateData> &statedatalist) {
    LOGI << "Total " << statedatalist.size() << " pose states from " << Logging::doubleData(statedatalist.begin()->time)
         << " to " << Logging::doubleData(statedatalist.back().time);

    for (auto &statedata : statedatalist) {
        // 位姿
        ceres::Manifold *manifold = new PoseManifold();
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), manifold);
        problem.AddParameterBlock(statedata.mix, Preintegration::numMixParameter(preintegration_options_));
    }
}

void Optimizer::addVisualParameters(ceres::Problem &problem) {
    if (visual_map_->keyframes().size() < 2) {
        return;
    }

    // 逆深度
    invdepthlist_.clear();
    for (const auto &landmark : visual_map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            // 是否为地图中的关键帧路标点
            auto frame = mappoint->referenceFrame();
            if (!frame || !visual_map_->isKeyFrameInMap(frame)) {
                continue;
            }

            double depth         = mappoint->depth();
            double inverse_depth = 1.0 / depth;

            // 确保深度数值有效
            if (std::isnan(inverse_depth)) {
                mappoint->setOutlier(true);
                LOGE << "Mappoint " << mappoint->id() << " is wrong with depth " << depth << " type "
                     << mappoint->mapPointType();
                continue;
            }

            invdepthlist_[mappoint->id()] = inverse_depth;
            problem.AddParameterBlock(&invdepthlist_[mappoint->id()], 1);

            // 增加优化次数计数
            mappoint->addOptimizedTimes();
        }
    }

    // 外参参数化
    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(extrinsic_b_c_, 7, manifold);

    // 固定外参
    if (!optimize_estimate_cam_extrinsic_ || !visual_map_->isWindowFull()) {
        problem.SetParameterBlockConstant(extrinsic_b_c_);
    }

    // 时间延时
    if (!optimize_estimate_cam_td_ || !visual_map_->isWindowFull()) {
        for (auto frame : visual_map_->keyframes()) {
            problem.AddParameterBlock(&frame.second->timeDelayEst(), 1);
            problem.SetParameterBlockConstant(&frame.second->timeDelayEst());
        }
    }
}

Residuals Optimizer::addReprojectionFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                            deque<StateData> &statedatalist) {
    Residuals residual_ids;

    if (visual_map_->keyframes().size() < 2) {
        return residual_ids;
    }

    // 鲁棒核函数
    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {
        loss_function = new ceres::HuberLoss(1.0);
    }

    // 所有路标点的观测
    residual_ids.clear();
    for (const auto &landmark : visual_map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        // 参考帧
        auto ref_frame = mappoint->referenceFrame();

        // 由于前端修改参考帧导致路标点暂时失效
        if (!visual_map_->isKeyFrameInMap(ref_frame)) {
            continue;
        }

        auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
        size_t ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        // 逆深度不能为0
        double *invdepth = &invdepthlist_[mappoint->id()];
        if (*invdepth == 0) {
            *invdepth = 1.0 / visual::VisualMapPoint::DEFAULT_DEPTH;
        }

        // 激光深度辅助, 深度约束因子
        if (is_use_lidar_depth_) {
            if (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED) {
                auto factor = DepthFactor::create(mappoint->lidarDepth(), visual::Tracking::ASSOCIATE_DEPTH_STD);
                problem.AddResidualBlock(factor, loss_function, invdepth);
            }
        }

        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        // 所有的观测
        for (const auto &observation : mappoint->observations()) {
            auto obs_feature = observation.lock();
            // 无效特征
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            // 非关键帧特征, 参考帧
            if (!obs_frame || !obs_frame->isKeyFrame() || !visual_map_->isKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            // 观测帧
            auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
            size_t obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());

            if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                     << Logging::doubleData(obs_frame->stamp());
                continue;
            }

            auto factor            = new ReprojectionFactor(ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(),
                                                            obs_feature->velocityInPixel(), ref_frame->timeDelay(),
                                                            obs_frame->timeDelay(), optimize_reprojection_error_std_);
            auto residual_block_id = problem.AddResidualBlock(
                factor, loss_function, statedatalist[ref_frame_index].pose, statedatalist[obs_frame_index].pose,
                extrinsic_b_c_, invdepth, &ref_frame->timeDelayEst(), &obs_frame->timeDelayEst());
            residual_ids.push_back(residual_block_id);
        }
    }

    if (is_use_prior_ && optimize_estimate_cam_extrinsic_) {
        // 对外参添加先验
        auto pose_factor = new ImuPosePriorFactor(extrinsic_cam_prior_, extrinsic_cam_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, extrinsic_b_c_);
    }

    if (optimize_estimate_cam_td_) {
        auto keyframeids = visual_map_->orderedKeyFrames();
        for (size_t k = 0; k < keyframeids.size() - 1; k++) {
            auto frame_pre = visual_map_->keyframes().at(keyframeids[k]);
            auto frame_cur = visual_map_->keyframes().at(keyframeids[k + 1]);
            auto factor    = new TimeDelayFactor(frame_cur->stamp() - frame_pre->stamp());
            problem.AddResidualBlock(factor, nullptr, &frame_pre->timeDelayEst(), &frame_cur->timeDelayEst());
        }
    }

    return residual_ids;
}

void Optimizer::addImuFactors(ceres::Problem &problem, deque<StateData> &statedatalist,
                              deque<PreBasePtr> &preintegrationlist) {
    for (size_t k = 0; k < preintegrationlist.size(); k++) {
        // 预积分因子
        auto factor = new PreintegrationFactor(preintegrationlist[k]);
        problem.AddResidualBlock(factor, nullptr, statedatalist[k].pose, statedatalist[k].mix,
                                 statedatalist[k + 1].pose, statedatalist[k + 1].mix);
    }

    // 添加IMU误差约束, 限制过大的误差估计
    auto factor = new ImuErrorFactor(preintegration_options_);
    problem.AddResidualBlock(factor, nullptr, statedatalist.back().mix);

    // IMU初始先验因子, 仅限于初始化
    if (is_use_prior_) {
        auto pose_factor = new ImuPosePriorFactor(pose_prior_, pose_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, statedatalist[0].pose);

        auto mix_factor = new ImuMixPriorFactor(preintegration_options_, mix_prior_, mix_prior_std_);
        problem.AddResidualBlock(mix_factor, nullptr, statedatalist[0].mix);
    }

    vector<double> average;
    if (MISC::detectZeroVelocity(preintegrationlist.back()->imuBuffer(), imudatarate_, average)) {
        LOGI << "Zero velocity is detected at " << Logging::doubleData(statedatalist.back().time);
        auto factor = new ImuZeroVelocityFactor(preintegration_options_, ZERO_VELOCITY_STD);
        problem.AddResidualBlock(factor, nullptr, statedatalist.back().mix);
    }
}

ResidualOutliers Optimizer::visualOutlierCullingByChi2(ceres::Problem &problem, vector<ResidualId> &residual_ids) {
    static double chi2_threshold = 5.991; // 0.05

    size_t residual_size = residual_ids.size();
    ResidualOutliers outlier_ids(residual_size);

    auto culling_function = [&](const tbb::blocked_range<size_t> &range) {
        for (size_t k = range.begin(); k != range.end(); k++) {
            const auto &residual_id = residual_ids[k];

            // 获取参数块和CostFunction
            vector<double *> parameter_blocks;
            problem.GetParameterBlocksForResidualBlock(residual_id, &parameter_blocks);
            const ceres::CostFunction *cost_function = problem.GetCostFunctionForResidualBlock(residual_id);

            // 计算残差
            Vector2d residual;
            cost_function->Evaluate(parameter_blocks.data(), residual.data(), nullptr);

            // 判断粗差
            bool is_outlier = false;
            double cost     = residual.transpose() * residual;
            if (cost > chi2_threshold) {
                is_outlier = true;
            }
            outlier_ids[k] = std::make_pair(residual_id, is_outlier);
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, residual_size), culling_function);

    return outlier_ids;
}

bool Optimizer::visualLandmarkOutlierCulling() {
    if (visual_map_->keyframes().size() < 2) {
        return false;
    }

    // 移除非关键帧中的路标点, 不能在遍历中直接移除, 否则破坏了遍历
    vector<visual::VisualMapPoint::Ptr> mappoints;
    int num_outliers_mappoint = 0;
    int num_outliers_feature  = 0;
    int num1 = 0, num2 = 0, num3 = 0;
    for (auto &landmark : visual_map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 未参与优化的无效路标点
        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        // 路标点在滑动窗口内的所有观测
        vector<double> errors;
        for (const auto &observation : mappoint->observations()) {
            auto feat = observation.lock();
            // 无效特征
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            // 非关键帧特征
            if (!frame || !frame->isKeyFrame() || !visual_map_->isKeyFrameInMap(frame)) {
                continue;
            }

            auto pp = feat->keyPoint();

            // 计算重投影误差
            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), pp).norm();

            // 大于3倍阈值, 则禁用当前观测
            if (!visual::VisualMapPoint::isGoodToTrack(camera_, pp, frame->pose(), mappoint->pos(),
                                                       3.0 * optimize_reprojection_std_, 1.0)) {
                feat->setOutlier(true);
                mappoint->decreaseUsedTimes();

                // 如果当前观测帧是路标点的参考帧, 直接设置为outlier
                if (frame == mappoint->referenceFrame()) {
                    mappoint->setOutlier(true);
                    mappoints.push_back(mappoint);
                    num_outliers_mappoint++;
                    num1++;
                    break;
                }
                num_outliers_feature++;
            } else {
                errors.push_back(error);
            }
        }

        // 有效观测不足, 平均重投影误差较大, 则为粗差
        if (errors.size() < 2) {
            mappoint->setOutlier(true);
            mappoints.push_back(mappoint);
            num_outliers_mappoint++;
            num2++;
        } else {
            double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
            if (avg_error > optimize_reprojection_std_) {
                mappoint->setOutlier(true);
                mappoints.push_back(mappoint);
                num_outliers_mappoint++;
                num3++;
            }
        }
    }

    // 移除outliers
    for (auto &mappoint : mappoints) {
        visual_map_->removeMappoint(mappoint);
    }

    LOGI << "Culled " << num_outliers_mappoint << " visual mappoint with " << num_outliers_feature
         << " bad observed features " << num1 << ", " << num2 << ", " << num3;

    outliers_[0] = num_outliers_mappoint;
    outliers_[1] = num_outliers_feature;

    return true;
}

void Optimizer::doReintegration(deque<PreBasePtr> &preintegrationlist, deque<StateData> &statedatalist) {
    int cnt = 0;
    for (size_t k = 0; k < preintegrationlist.size(); k++) {
        IntegrationState state = Preintegration::stateFromData(statedatalist[k], preintegration_options_);
        Vector3d dbg           = preintegrationlist[k]->deltaState().bg - state.bg;
        Vector3d dba           = preintegrationlist[k]->deltaState().ba - state.ba;
        if ((dbg.norm() > 6 * integration_parameters_->gyr_bias_std) ||
            (dba.norm() > 6 * integration_parameters_->acc_bias_std)) {
            preintegrationlist[k]->reintegration(state);
            cnt++;
        }
    }
    if (cnt) {
        LOGW << "Reintegration " << cnt << " preintegration";
    }
}

int Optimizer::getStateDataIndex(const deque<double> &timelist, double time) {

    size_t index = MISC::getStateDataIndex(timelist, time, MISC::MINIMUM_TIME_INTERVAL);
    if (!MISC::isTheSameTimeNode(timelist[index], time, MISC::MINIMUM_TIME_INTERVAL)) {
        LOGW << "Wrong matching time node " << Logging::doubleData(timelist[index]) << " to "
             << Logging::doubleData(time);
        return -1;
    }
    return static_cast<int>(index);
}

void Optimizer::constructPrior(const StateData &statedata, bool is_zero_velocity) {
    // 初始先验
    double pos_prior_std  = 0.1;                                       // 0.1 m
    double att_prior_std  = 0.1 * D2R;                                 // 0.1 deg
    double vel_prior_std  = 0.1;                                       // 0.1 m/s
    double bg_prior_std   = integration_parameters_->gyr_bias_std * 3; // Bias std * 3
    double ba_prior_std   = ab_prior_std_;
    double sodo_prior_std = 0.005;     // 5000 PPM
    double avb_prior_std  = 0.5 * D2R; // 0.5 deg

    if (!is_zero_velocity) {
        att_prior_std = 0.5 * D2R; // 0.5 deg
        bg_prior_std  = gb_prior_std_;
        vel_prior_std = 0.5;
    }

    memcpy(pose_prior_, statedata.pose, sizeof(double) * 7);
    memcpy(mix_prior_, statedata.mix, sizeof(double) * 18);
    for (size_t k = 0; k < 3; k++) {
        pose_prior_std_[k + 0] = pos_prior_std;
        pose_prior_std_[k + 3] = att_prior_std;

        mix_prior_std_[k + 0] = vel_prior_std;
        mix_prior_std_[k + 3] = bg_prior_std;
        mix_prior_std_[k + 6] = ba_prior_std;
    }

    // 里程计模式
    pose_prior_std_[0] = 0.001;
    pose_prior_std_[1] = 0.001;
    pose_prior_std_[2] = 0.001;
    pose_prior_std_[5] = 0.001 * D2R;

    mix_prior_std_[9]  = sodo_prior_std;
    mix_prior_std_[10] = avb_prior_std;
    mix_prior_std_[11] = avb_prior_std;

    if (optimize_estimate_cam_extrinsic_) {
        // 外参先验
        if (optimize_cam_extrinsic_accurate_) {
            pos_prior_std = 0.05;      // 0.05 m
            att_prior_std = 0.1 * D2R; // 0.1 deg
        } else {
            pos_prior_std = 0.1;       // 0.1 m
            att_prior_std = 1.0 * D2R; // 1.0 deg
        }

        memcpy(extrinsic_cam_prior_, extrinsic_b_c_, sizeof(double) * 7);
        for (size_t k = 0; k < 3; k++) {
            extrinsic_cam_prior_std_[k + 0] = pos_prior_std;
            extrinsic_cam_prior_std_[k + 3] = att_prior_std;
        }
    }

    is_use_prior_ = true;
}

const vector<double> &Optimizer::reprojectionError() {
    // 路标点重投影误差统计
    vector<double> reprojection_errors;
    for (auto &landmark : visual_map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 未参与优化的无效路标点
        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        // 所有的观测
        vector<double> errors;
        for (const auto &observation : mappoint->observations()) {
            auto feat = observation.lock();
            // 无效特征
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            // 非关键帧特征
            if (!frame || !frame->isKeyFrame() || !visual_map_->isKeyFrameInMap(frame)) {
                continue;
            }

            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), feat->keyPoint()).norm();
            errors.push_back(error);
        }
        if (errors.empty()) {
            LOGE << "Mappoint " << mappoint->id() << " with zero observation";
            continue;
        }
        double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        reprojection_errors.emplace_back(avg_error);
    }

    // 避免为空造成内存错误
    if (reprojection_errors.empty()) {
        reprojection_errors.push_back(0);
    }
    reprojection_.clear();

    // 最小误差
    double min_error = *std::min_element(reprojection_errors.begin(), reprojection_errors.end());
    reprojection_.push_back(min_error);
    // 最大误差
    double max_error = *std::max_element(reprojection_errors.begin(), reprojection_errors.end());
    reprojection_.push_back(max_error);
    // 平均误差
    double avg_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) /
                       static_cast<double>(reprojection_errors.size());
    reprojection_.push_back(avg_error);
    // RMSE
    double sq_sum =
        std::inner_product(reprojection_errors.begin(), reprojection_errors.end(), reprojection_errors.begin(), 0.0);
    double rms_error = std::sqrt(sq_sum / static_cast<double>(reprojection_errors.size()));
    reprojection_.push_back(rms_error);

    return reprojection_;
}

void Optimizer::updateCameraExtrinsic(Pose &pose_b_c, double &td_b_c) const {
    pose_b_c.t[0] = extrinsic_b_c_[0];
    pose_b_c.t[1] = extrinsic_b_c_[1];
    pose_b_c.t[2] = extrinsic_b_c_[2];

    Quaterniond q_b_c = Quaterniond(extrinsic_b_c_[6], extrinsic_b_c_[3], extrinsic_b_c_[4], extrinsic_b_c_[5]);
    pose_b_c.R        = Rotation::quaternion2matrix(q_b_c.normalized());

    td_b_c = extrinsic_b_c_[7];
}

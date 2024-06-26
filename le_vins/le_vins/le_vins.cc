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

#include "le_vins/le_vins.h"

#include "common/earth.h"
#include "common/gpstime.h"
#include "common/logging.h"
#include "common/misc.h"
#include "common/timecost.h"
#include "lidar/pointcloud.h"

#include <yaml-cpp/yaml.h>

VINS::VINS(const string &configfile, const string &outputpath, visual::VisualDrawer::Ptr visual_drawer) {
    // 初始状态
    vinsstate_ = VINS_ERROR;

    // 加载配置
    YAML::Node config;
    std::vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    // 文件IO
    visual_pts_filesaver_     = FileSaver::create(outputpath + "/visual_mappoint.txt", 3);
    visual_stat_filesaver_    = FileSaver::create(outputpath + "/visual_statistic.txt", 3);
    cam_ext_filesaver_        = FileSaver::create(outputpath + "/visual_extrinsic.txt", 7);
    imu_err_filesaver_        = FileSaver::create(outputpath + "/VINS_IMU_ERR.bin", 7, FileSaver::BINARY);
    traj_filesaver_           = FileSaver::create(outputpath + "/trajectory.csv", 8);

    // make a copy of configuration file
    std::ofstream ofconfig(outputpath + "/le_vins.yaml");
    ofconfig << YAML::Dump(config);
    ofconfig.close();

    // 积分配置参数
    integration_parameters_ = std::make_shared<IntegrationParameters>();

    // IMU噪声参数
    integration_parameters_->gyr_arw      = config["imu"]["arw"].as<double>() * D2R / 60.0;
    integration_parameters_->gyr_bias_std = config["imu"]["gbstd"].as<double>() * D2R / 3600.0;
    integration_parameters_->acc_vrw      = config["imu"]["vrw"].as<double>() / 60.0;
    integration_parameters_->acc_bias_std = config["imu"]["abstd"].as<double>() * 1.0e-5;
    integration_parameters_->corr_time    = config["imu"]["corrtime"].as<double>() * 3600;

    // IMU数据率
    imudatarate_ = config["imu"]["imudatarate"].as<double>();
    imudatadt_   = 1.0 / imudatarate_;

    // 里程计参数
    integration_parameters_->gravity = NORMAL_GRAVITY;
    integration_config_.gravity      = {0, 0, integration_parameters_->gravity};
    integration_config_.iswithearth  = false;
    integration_config_.isuseodo     = false;
    preintegration_options_          = Preintegration::getOptions(integration_config_);

    // 相机参数
    is_use_lidar_depth_       = config["visual"]["is_use_lidar_depth"].as<bool>();
    vector<double> intrinsic  = config["visual"]["intrinsic"].as<std::vector<double>>();
    vector<double> distortion = config["visual"]["distortion"].as<std::vector<double>>();
    vector<int> resolution    = config["visual"]["resolution"].as<std::vector<int>>();

    camera_ = visual::Camera::createCamera(intrinsic, distortion, resolution);

    // Camera外参
    vecdata           = config["visual"]["q_b_c"].as<std::vector<double>>();
    Quaterniond q_b_c = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["visual"]["t_b_c"].as<std::vector<double>>();
    Vector3d t_b_c    = Eigen::Vector3d(vecdata.data());
    td_b_c_           = config["visual"]["td_b_c"].as<double>();

    pose_b_c_.R = q_b_c.toRotationMatrix();
    pose_b_c_.t = t_b_c;

    // 雷达相机外参, 固定不调整, 用于激光深度增强
    vecdata           = config["visual"]["q_c_l"].as<std::vector<double>>();
    Quaterniond q_c_l = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["visual"]["t_c_l"].as<std::vector<double>>();
    Vector3d t_c_l    = Eigen::Vector3d(vecdata.data());
    pose_c_l_.R       = q_c_l.toRotationMatrix();
    pose_c_l_.t       = t_c_l;
    td_b_l_           = config["visual"]["td_b_l"].as<double>();

    // 保留的缓存帧数量
    reserved_buffer_counts_ = RESERVED_BUFFER_LENGTH / 0.1;

    // 可视化
    is_use_visualization_ = config["is_use_visualization"].as<bool>();

    optimize_estimate_cam_extrinsic_ = config["optimizer"]["optimize_estimate_cam_extrinsic"].as<bool>();
    optimize_estimate_cam_td_        = config["optimizer"]["optimize_estimate_cam_td"].as<bool>();
    size_t optimize_window_size      = config["optimizer"]["optimize_window_size"].as<size_t>();

    // 清理容器
    preintegrationlist_.clear();
    statedatalist_.clear();
    timelist_.clear();

    visual_map_    = std::make_shared<visual::VisualMap>(optimize_window_size);
    visual_drawer_ = std::move(visual_drawer);
    if (is_use_visualization_) {
        visual_drawer_thread_ = std::thread(&visual::VisualDrawer::run, visual_drawer_);
    }
    tracking_ = std::make_shared<visual::Tracking>(camera_, visual_map_, visual_drawer_, configfile, outputpath);

    // 优化对象
    optimizer_ =
        std::make_shared<Optimizer>(configfile, integration_parameters_, preintegration_options_, visual_map_, camera_);
    fusion_thread_ = std::thread(&VINS::runFusion, this);

    // 更新系统状态, 进入初始化状态
    vinsstate_ = VINS_INITIALIZING;
}

bool VINS::addNewImu(const IMU &imu) {
    if (imu_buffer_mutex_.try_lock()) {
        imu_buffer_.push(imu);
        imu_buffer_mutex_.unlock();
        return true;
    }

    return false;
}

bool VINS::addNewVisualFrame(const VisualFrame::Ptr &frame) {
    Lock lock(visual_frame_buffer_mutex_);
    visual_frame_buffer_.push(frame);

    return true;
}

bool VINS::addNewPointCloud(PointCloudCustomPtr pointcloud) {
    // 激光深度增强
    if (is_use_lidar_depth_) {
        Lock lock(lidar_point_buffer_mutex_);

        // 拷贝点云数据
        lidar_point_buffer_.insert(lidar_point_buffer_.end(), pointcloud->points.begin(), pointcloud->points.end());
        double reserved_time = RESERVED_BUFFER_LENGTH + LIDAR_DEPTH_POINT_STACK_INTERVAL * 2;
        while (fabs(lidar_point_buffer_.back().time - lidar_point_buffer_.front().time) > reserved_time) {
            lidar_point_buffer_.pop_front();
        }
    }

    return true;
}

bool VINS::canAddData() {
    // Buffer容量为空则可以继续加入数据, 注意考虑IMU数据需要超前
    return visual_frame_buffer_.size() < reserved_buffer_counts_;
}

bool VINS::isBufferEmpty() {
    return imu_buffer_.empty() || visual_frame_buffer_.empty();
}

void VINS::setFinished() {
    is_finished_ = true;

    if (is_use_visualization_) {
        // 结束绘图线程
        visual_drawer_->setFinished();
        visual_drawer_thread_.join();
    }
    task_group_.wait();
    fusion_thread_.join();

    if (optimize_estimate_cam_extrinsic_ || optimize_estimate_cam_td_) {
        Quaterniond q_b_c = Rotation::matrix2quaternion(pose_b_c_.R);
        Vector3d t_b_c    = pose_b_c_.t;
        LOGW << "Estimated camera extrinsics: "
             << absl::StrFormat("(%0.6lf, %0.6lf, %0.6lf, %0.6lf), (%0.3lf, %0.3lf, %0.3lf), %0.4lf", q_b_c.x(),
                                q_b_c.y(), q_b_c.z(), q_b_c.w(), t_b_c.x(), t_b_c.y(), t_b_c.z(), td_b_c_);
    }
    LOGW << "VINS has finished processing";
}

void VINS::runFusion() {
    VisualFrame::Ptr visual_frame = nullptr;
    bool is_need_optimization     = false;
    double last_update_time       = 0;

    LOGI << "Fusion thread is started";
    while (!is_finished_) {
        if ((vinsstate_ == VINS_INITIALIZING)) {
            while (!is_finished_ && imu_buffer_.empty()) {
                usleep(WAITING_DELAY_IN_US);
            }
        } else {
            // 等待图像数据
            while (!is_finished_ && visual_frame_buffer_.empty()) {
                usleep(WAITING_DELAY_IN_US);
            }

            // 取最新的关键帧
            if (!is_finished_) {
                Lock lock(visual_frame_buffer_mutex_);
                visual_frame = visual_frame_buffer_.front();
                visual_frame_buffer_.pop();
            }
        }
        if (is_finished_) {
            break;
        }
        if (visual_frame_buffer_.size() > reserved_buffer_counts_) {
            LOGI << "Buffer size: " << visual_frame_buffer_.size();
        }

        // 融合状态
        if (vinsstate_ == VINS_INITIALIZING) {
            // 移除过多的INS状态
            while (ins_window_.size() > MAXIMUM_INS_NUMBER) {
                ins_window_.pop_front();
            }

            // 初始时刻
            if (ins_window_.empty()) {
                Lock lock(imu_buffer_mutex_);
                auto imu = imu_buffer_.front();

                // 向前取整, 取下一个有效时刻
                last_update_time = ceil(imu.time / INS_INITIALIZATION_INTERVAL) * INS_INITIALIZATION_INTERVAL;
            }

            // 等待IMU数据有效并加入INS窗口
            double current_time = last_update_time + INS_INITIALIZATION_INTERVAL;
            if (!waitImuAddToInsWindow(current_time)) {
                break;
            }

            // 初始化参数, 第一个节点
            if (vinsInitialization(last_update_time, current_time)) {
                vinsstate_ = VINS_INITIALIZING_VISUAL;
            }
            last_update_time = current_time;
        } else if (vinsstate_ >= VINS_INITIALIZING_VISUAL) {

            // 视觉跟踪处理
            if (vinsVisualTrackingProcessing(visual_frame) == 1) {

                // 插入关键帧节点
                addNewVisualFrameTimeNode(visual_frame);

                // 初始帧不优化
                if (vinsstate_ == VINS_INITIALIZING_VISUAL) {
                    is_need_optimization = false;

                    // VIO初始化完成
                    vinsstate_ = VINS_NORMAL;
                } else if (vinsstate_ == VINS_NORMAL) {
                    is_need_optimization = true;
                }
            }
        }

        if (is_need_optimization) {
            is_need_optimization = false;

            // 优化求解
            vinsOptimizationProcessing();

            // 更新窗口内状态
            auto state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
            MISC::redoInsMechanization(integration_config_, state, RESERVED_INS_NUM, ins_window_);

            // 可视化
            if (is_use_visualization_) {
                task_group_.run([this]() { vinsVisualVisualization(); });
            }
        }

    } // while
    LOGI << "Fusion thread is exited";
}

bool VINS::vinsOptimizationProcessing() {

    // 两次非线性优化并进行粗差剔除
    optimizer_->optimization(timelist_, statedatalist_, preintegrationlist_);

    if (visual_map_->isWindowFull()) {
        // 移除所有窗口中间插入的非关键帧
        removeNonKeyFrame();
    }

    // 边缘化, 移除旧的观测, 按时间对齐到保留的最后一个关键帧
    if (visual_map_->isWindowFull()) {
        optimizer_->marginalization(timelist_, statedatalist_, preintegrationlist_);
    }

    // 更新外参参数
    updateParametersFromOptimizer();

    // 统计并输出视觉相关的参数
    parametersStatistic();

    return true;
}

bool VINS::vinsVisualVisualization() {
    // 当前窗口内的路标点
    const auto &landmarks = visual_map_->landmarks();
    vector<Vector3d> current;
    for (const auto &landmark : landmarks) {
        const auto &mappoint = landmark.second;
        if (mappoint && !mappoint->isOutlier()) {
            current.push_back(mappoint->pos());
        }
    }
    if (!current.empty()) {
        visual_drawer_->updateCurrentMappoints(current);
    }

    // 新的固定的路标点
    const auto &fixed = optimizer_->newFixedVisualMappoints();
    if (!fixed.empty()) {
        visual_drawer_->updateNewFixedMappoints(fixed);
    }

    // 位姿可视化
    auto state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
    Pose pose;
    pose.t = state.p;
    pose.R = state.q.toRotationMatrix();

    visual_drawer_->updateMap(pose);

    return true;
}

int VINS::vinsVisualTrackingProcessing(VisualFrame::Ptr frame) {
    static PointCloud pointcloud;

    // 时标更新
    frame->setTimeDelay(td_b_c_);
    frame->setStamp(frame->stamp() + td_b_c_);

    // 进行机械编排处理
    if (!waitImuDoInsMechanization(frame->stamp())) {
        return -1;
    }

    TimeCost timecost;

    // 获取初始位姿
    Pose pose;
    MISC::getPoseFromInsWindow(ins_window_, pose_b_c_, frame->stamp(), pose);
    frame->setPose(pose);

    auto trackstate = tracking_->track(frame, pointcloud);
    if (trackstate == visual::TRACK_LOST) {
        LOGE << "Tracking lost at " << Logging::doubleData(frame->stamp());
    }

    // 包括第一帧在内的所有关键帧, 跟踪失败时的当前帧也会成为新的关键帧
    if (tracking_->isNewKeyFrame() || (trackstate == visual::TRACK_FIRST_FRAME)) {
        if (is_use_lidar_depth_ && (frame->keyFrameState() != visual::KEYFRAME_REMOVE_SECOND_NEW)) {
            // 点云数据投影准备
            pointcloud.clear();

            // 点云数据投影到参考帧, 在上一个关键帧后的时间进行计算
            if (!lidar_point_buffer_.empty() && (vinsstate_ == VINS_NORMAL)) {
                pointcloudStackForVisualTracking(frame, pointcloud);
            }
        }

        stat_visual_tracking_cost_ = timecost.costInMillisecond();
        LOGI << "Tracking cost " << stat_visual_tracking_cost_ << " ms";
        return 1;
    }

    return 0;
}

bool VINS::pointcloudStackForVisualTracking(const VisualFrame::Ptr frame, PointCloud &pointcloud) {
    TimeCost timecost;

    double last_stamp = frame->stamp() - LIDAR_DEPTH_POINT_STACK_INTERVAL;
    double curr_stamp = frame->stamp();

    while (!is_finished_) {
        lidar_point_buffer_mutex_.lock();
        auto point_stamp = lidar_point_buffer_.back().time;
        lidar_point_buffer_mutex_.unlock();
        if (point_stamp > curr_stamp) {
            break;
        }

        usleep(WAITING_DELAY_IN_US);
    }

    // 点云数据加锁
    Lock lock(lidar_point_buffer_mutex_);

    // 移除过期的点云
    while (true) {
        if (lidar_point_buffer_.empty()) {
            break;
        }

        auto &point = lidar_point_buffer_.front();
        if (point.time < last_stamp) {
            lidar_point_buffer_.pop_front();
        } else {
            break;
        }
    }

    if (lidar_point_buffer_.empty()) {
        LOGW << "No valid point cloud in current frame " << Logging::doubleData(curr_stamp);
        return false;
    }

    size_t num_projected = lidar_point_buffer_.size();
    for (size_t k = 0; k < lidar_point_buffer_.size(); k++) {
        if (lidar_point_buffer_[k].time > curr_stamp) {
            num_projected = k;
            break;
        }
    }
    pointcloud.points.resize(num_projected);

    // 使用连续的INS位姿
    Pose last_keyframe_pose;
    MISC::getPoseFromInsWindow(ins_window_, pose_b_c_, curr_stamp, last_keyframe_pose);

    // 有效点云数据投影
    auto projection_function = [&](const tbb::blocked_range<size_t> &range) {
        for (size_t k = range.begin(); k != range.end(); k++) {
            auto &raw = lidar_point_buffer_[k];
            auto norm = raw.getVector3fMap().norm();
            if ((norm > visual::VisualMapPoint::FARTHEST_DEPTH) || (norm < visual::VisualMapPoint::NEAREST_DEPTH)) {
                continue;
            }

            // 获取点对应时刻的相机位姿, 考虑IMU数据延时
            Pose pose;
            MISC::getPoseFromInsWindow(ins_window_, pose_b_c_, raw.time + td_b_l_, pose);

            // 通过激光和相机外参转换到相机坐标系
            PointType point;
            point.getVector3fMap() = raw.getVector3fMap();
            pointcloud[k] =
                lidar::PointCloudCommon::relativeProjectionToCamera(pose_c_l_, last_keyframe_pose, pose, point);
        }
    };
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_projected), projection_function);

    LOGI << "Projection to " << Logging::doubleData(curr_stamp) << " cost " << timecost.costInMillisecond()
         << " ms with size " << pointcloud.size();

    return !pointcloud.empty();
}

bool VINS::waitImuAddToInsWindow(double time) {
    double end_time = time + imudatadt_ * 2;

    // 等待IMU数据
    waitImuData(end_time);

    // IMU数据加入窗口
    Lock lock(imu_buffer_mutex_);
    while (true) {
        if (imu_buffer_.empty() || (imu_buffer_.front().time > end_time)) {
            break;
        }
        auto imu = imu_buffer_.front();
        imu_buffer_.pop();

        ins_window_.emplace_back(imu, IntegrationState());
    }

    return !is_finished_;
}

bool VINS::waitImuDoInsMechanization(double time) {
    double end_time = time + imudatadt_ * 2;

    // 等待IMU数据
    waitImuData(end_time);

    // INS机械编排
    doInsMechanization(end_time);

    return !is_finished_;
}

bool VINS::waitImuData(double time) {
    while (true) {
        imu_buffer_mutex_.lock();
        if ((!imu_buffer_.empty() && (imu_buffer_.back().time < time)) ||
            (imu_buffer_.empty() && !ins_window_.empty() && ins_window_.back().first.time < time)) {
            imu_buffer_mutex_.unlock();

            usleep(WAITING_DELAY_IN_US);
        } else {
            imu_buffer_mutex_.unlock();
            break;
        }

        if (is_finished_) {
            break;
        }
    }

    return true;
}

void VINS::doInsMechanization(double time) {
    // INS机械编排到最新IMU数据
    auto imu_pre = ins_window_.back().first;
    auto state   = ins_window_.back().second;

    write_state_mutex_.lock();
    write_state_buffer_.clear();
    Lock lock(imu_buffer_mutex_);
    while (true) {
        if (imu_buffer_.empty()) {
            break;
        }

        auto imu_cur = imu_buffer_.front();
        imu_buffer_.pop();

        MISC::insMechanization(integration_config_, imu_pre, imu_cur, state);
        ins_window_.push_back({imu_cur, state});

        // 保存定位结果
        write_state_buffer_.push_back(state);

        imu_pre = imu_cur;

        // 总是积分到大于设定时刻
        if (imu_cur.time > time) {
            break;
        }
    }
    write_state_mutex_.unlock();

    // 写入文件
    task_group_.run([this]() {
        Lock lock(write_state_mutex_);
        for (const auto &state : write_state_buffer_) {
            writeNavResult(state);
        }
    });
}

void VINS::addNewVisualFrameTimeNode(VisualFrame::Ptr frame) {
    double frametime = frame->stamp();

    // 添加关键帧
    LOGI << "Insert keyframe " << frame->keyFrameId() << " at " << Logging::doubleData(frame->stamp()) << " with "
         << frame->unupdatedMappoints().size() << " new mappoints";
    visual_map_->insertKeyFrame(frame);

    addNewTimeNode(frametime, NODE_VISUAL);

    // 移除多余的预积分节点
    removeUnusedTimeNode();
}

bool VINS::removeUnusedTimeNode() {
    if (unused_time_node_ == 0) {
        return false;
    }

    // 移除多余的节点
    int index = optimizer_->getStateDataIndex(timelist_, unused_time_node_);

    // Exception
    if (index < 0) {
        return false;
    }

    auto first_preintegration  = preintegrationlist_[index - 1];
    auto second_preintegration = preintegrationlist_[index];
    auto imu_buffer            = second_preintegration->imuBuffer();

    // 将后一个预积分的IMU数据合并到前一个, 不包括第一个IMU数据
    for (size_t k = 1; k < imu_buffer.size(); k++) {
        first_preintegration->addNewImu(imu_buffer[k]);
    }

    // 移除时间节点, 以及后一个预积分
    preintegrationlist_.erase(preintegrationlist_.begin() + index);
    timelist_.erase(timelist_.begin() + index);
    statedatalist_.erase(statedatalist_.begin() + index);

    LOGI << "Remove unused time node " << Logging::doubleData(unused_time_node_);
    unused_time_node_ = 0;

    return true;
}

IntegrationState VINS::addNewTimeNode(double time, NodeType type) {

    vector<IMU> series;
    IntegrationState state;

    // 获取时段内用于预积分的IMU数据
    double start = timelist_.back();
    double end   = time;
    MISC::getImuSeriesFromTo(ins_window_, start, end, series);

    state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);

    // 新建立新的预积分
    {
        preintegrationlist_.emplace_back(
            Preintegration::createPreintegration(integration_parameters_, series[0], state, preintegration_options_));
    }

    // 预积分, 从第二个历元开始
    for (size_t k = 1; k < series.size(); k++) {
        preintegrationlist_.back()->addNewImu(series[k]);
    }

    // 当前状态加入到滑窗中
    state      = preintegrationlist_.back()->currentState();
    state.time = time;
    statedatalist_.emplace_back(Preintegration::stateToData(state, preintegration_options_));

    // 新的时间节点
    LOGI << "Add new " << NODE_NAME.at(type) << " time node at "
         << absl::StrFormat("%0.6lf with dt %0.3lf", time, time - timelist_.back());
    timelist_.push_back(time);

    return state;
}

void VINS::parametersStatistic() {

    // 待输出的参数
    vector<double> parameters;

    parameters.clear();

    // 所有关键帧
    const auto keyframeids = visual_map_->orderedKeyFrames();
    size_t size            = keyframeids.size();
    if (size < 2) {
        return;
    }
    const auto &keyframes = visual_map_->keyframes();

    // 最新的关键帧
    auto frame_cur = keyframes.at(keyframeids[size - 1]);
    auto frame_pre = keyframes.at(keyframeids[size - 2]);

    // 时间戳
    parameters.push_back(frame_cur->stamp());
    parameters.push_back(frame_cur->stamp() - frame_pre->stamp());

    // 当前关键帧与上一个关键帧的id差, 即最新关键帧的跟踪帧数
    auto frame_cnt = static_cast<double>(frame_cur->id() - frame_pre->id());
    parameters.push_back(frame_cnt);

    // 特征点数量
    parameters.push_back(static_cast<double>(frame_cur->numFeatures()));

    // 重投影误差统计
    const auto &reprojection = optimizer_->reprojectionError();
    parameters.insert(parameters.end(), reprojection.begin(), reprojection.end());

    // 迭代次数
    const auto &iteration = optimizer_->iterations();
    parameters.insert(parameters.end(), iteration.begin(), iteration.end());

    // 计算耗时
    const auto &timecost = optimizer_->timecosts();
    parameters.insert(parameters.end(), timecost.begin(), timecost.end());

    // 路标点粗差
    const auto &outliers = optimizer_->outliers();
    parameters.push_back(outliers[0]); // mappoints
    parameters.push_back(outliers[1]); // features

    // 激光深度特征统计
    int num_mappoint_total = 0;
    int num_mappoint_depth = 0;
    for (const auto &feature : frame_cur->features()) {
        auto mappoint = feature.second->getMapPoint();
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        num_mappoint_total++;
        if (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED) {
            num_mappoint_depth++;
        }
    }

    parameters.push_back(num_mappoint_total);
    parameters.push_back(num_mappoint_depth);

    parameters.push_back(stat_visual_tracking_cost_);
    parameters.push_back(tracking_->trackingTimeCost());

    // 保存数据
    visual_stat_filesaver_->dump(parameters);
    visual_stat_filesaver_->flush();
}

void VINS::updateParametersFromOptimizer() {
    // 更新外参
    if (optimize_estimate_cam_td_ || optimize_estimate_cam_extrinsic_) {
        optimizer_->updateCameraExtrinsic(pose_b_c_, td_b_c_);

        vector<double> extrinsic;
        Vector3d euler = Rotation::matrix2euler(pose_b_c_.R) * R2D;

        extrinsic.push_back(timelist_.back());
        extrinsic.push_back(pose_b_c_.t[0]);
        extrinsic.push_back(pose_b_c_.t[1]);
        extrinsic.push_back(pose_b_c_.t[2]);
        extrinsic.push_back(euler[0]);
        extrinsic.push_back(euler[1]);
        extrinsic.push_back(euler[2]);
        extrinsic.push_back(td_b_c_);

        cam_ext_filesaver_->dump(extrinsic);
        cam_ext_filesaver_->flush();
    }

    // 输出路标点
    const auto &points = optimizer_->newFixedVisualMappoints();
    for (const auto &point : points) {
        visual_pts_filesaver_->dump({point.x(), point.y(), point.z()});
    }
}

bool VINS::removeNonKeyFrame() {
    auto keyframeids = visual_map_->orderedKeyFrames();

    auto newest_id    = keyframeids.back();
    auto visual_frame = visual_map_->keyframes().find(newest_id)->second;

    // 移除非关键帧
    if (visual_frame->keyFrameState() == visual::KEYFRAME_REMOVE_SECOND_NEW) {
        LOGI << "Remove nonkeyframe at " << Logging::doubleData(visual_frame->stamp());

        unused_time_node_ = visual_frame->stamp();

        // 仅需要重置关键帧标志, 从地图中移除次新关键帧即可, 无需调整状态参数和路标点
        visual_frame->resetKeyFrame();
        visual_map_->removeKeyFrame(visual_frame, false);
    }

    return true;
}

bool VINS::vinsInitialization(double last_time, double current_time) {

    if (ins_window_.size() < imudatarate_) {
        return false;
    }

    // 缓存数据用于零速检测
    vector<IMU> imu_buff;
    for (const auto &ins : ins_window_) {
        auto &imu = ins.first;
        if ((imu.time > last_time) && (imu.time < current_time)) {
            imu_buff.push_back(imu);
        }
    }

    // 零速检测估计陀螺零偏和横滚俯仰角
    vector<double> average;
    static Vector3d bg{0, 0, 0};
    static Vector3d attitude{0, 0, 0};
    static bool is_has_zero_velocity = false;

    bool is_zero_velocity = MISC::detectZeroVelocity(imu_buff, imudatarate_, average);
    if (is_zero_velocity) {
        // 陀螺零偏
        bg = Vector3d(average[0], average[1], average[2]);
        bg *= imudatarate_;

        // 重力调平获取横滚俯仰角
        Vector3d fb(average[3], average[4], average[5]);
        fb *= imudatarate_;

        attitude[0] = -asin(fb[1] / integration_parameters_->gravity);
        attitude[1] = asin(fb[0] / integration_parameters_->gravity);

        LOGI << "Zero velocity get gyroscope bias " << bg.transpose() * 3600 * R2D << ", roll " << attitude[0] * R2D
             << ", pitch " << attitude[1] * R2D;
        is_has_zero_velocity = true;
    }

    // 里程计速度大于MINMUM_ALIGN_VELOCITY, 或者非零速状态
    Vector3d position{0, 0, 0};
    Vector3d velocity{0, 0, 0};
    double initial_time = last_time;

    // 推算当前时刻速度, 避免错误初始化
    double current_velocity;
    {
        auto state = IntegrationState{
            .time = initial_time,
            .p    = position,
            .q    = Rotation::euler2quaternion(attitude),
            .v    = velocity,
            .bg   = bg,
            .ba   = {0, 0, 0},
            .sodo = 0.0,
            .avb  = {integration_parameters_->abv[1], integration_parameters_->abv[2]},
        };
        MISC::redoInsMechanization(integration_config_, state, RESERVED_INS_NUM * 2, ins_window_);
        current_velocity = ins_window_.back().second.v.norm();
    }

    if (!is_zero_velocity && (current_velocity > 0.2)) {

        // 速度矢量
        if (integration_config_.isuseodo) {
            // 轮速有效则使用轮速
            double odovel = imu_buff.back().odovel / imu_buff.back().dt;

            size_t start_index =
                ins_window_.size() - static_cast<size_t>(imudatarate_ * 1.1 * INS_INITIALIZATION_INTERVAL);
            for (size_t k = start_index; k < ins_window_.size(); k++) {
                const auto &imu = ins_window_[k].first;
                if (imu.time >= last_time) {
                    odovel = imu.odovel / imu.dt;
                    break;
                }
            }

            Matrix3d cbn = Rotation::euler2matrix(attitude);
            Matrix3d cbv = Rotation::euler2matrix(integration_parameters_->abv);
            velocity     = cbn * cbv.transpose() * Vector3d(odovel, 0, 0);
        } else {
            if (!is_has_zero_velocity) {
                // 直接重力调平获取横滚俯仰角
                Vector3d fb(average[3], average[4], average[5]);
                fb *= imudatarate_;

                attitude[0] = -asin(fb[1] / integration_parameters_->gravity);
                attitude[1] = asin(fb[0] / integration_parameters_->gravity);

                LOGW << "Get roll " << attitude[0] * R2D << ", pitch " << attitude[1] * R2D << " without zero velocity";
            }

            // 无轮速直接初始化
            position = Vector3d::Zero();
            velocity = Vector3d::Zero();
        }
    } else {
        // 移除无效的图像数据
        VisualFrame::Ptr frame;
        while (!is_finished_ && !visual_frame_buffer_.empty()) {
            visual_frame_buffer_mutex_.lock();
            frame = visual_frame_buffer_.front();
            visual_frame_buffer_mutex_.unlock();
            if (frame->stamp() > current_time) {
                break;
            }
            visual_frame_buffer_.pop();
        }

        // 零速状态返回
        return false;
    }

    // 初始状态, 从上一秒开始
    auto state = IntegrationState{
        .time = initial_time,
        .p    = position,
        .q    = Rotation::euler2quaternion(attitude),
        .v    = velocity,
        .bg   = bg,
        .ba   = {0, 0, 0},
        .sodo = 0.0,
        .avb  = {integration_parameters_->abv[1], integration_parameters_->abv[2]},
    };

    // 初始时间节点
    statedatalist_.emplace_back(Preintegration::stateToData(state, preintegration_options_));
    timelist_.push_back(initial_time);

    // 初始先验
    optimizer_->constructPrior(statedatalist_.front(), is_has_zero_velocity);

    // 计算第一秒的INS结果
    state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
    MISC::redoInsMechanization(integration_config_, state, RESERVED_INS_NUM, ins_window_);
    {
        // 输出第一秒的结果
        write_state_mutex_.lock();
        write_state_buffer_.clear();
        for (size_t k = 0; k < ins_window_.size(); k++) {
            if (ins_window_[k].first.time >= initial_time) {
                write_state_buffer_.push_back(ins_window_[k].second);
            }
        }
        write_state_mutex_.unlock();

        // 写入文件
        task_group_.run([this]() {
            Lock lock(write_state_mutex_);
            for (const auto &state : write_state_buffer_) {
                writeNavResult(state);
            }
        });
    }

    start_time_ = current_time;
    LOGI << "Initialization at " << Logging::doubleData(last_time);
    // 当前时间节点
    addNewTimeNode(current_time, NODE_IMU);

    // 同步图像数据, 必须有图像数据

    // 等待图像数据有效
    VisualFrame::Ptr frame;
    while (!is_finished_) {
        visual_frame_buffer_mutex_.lock();
        frame = visual_frame_buffer_.back();
        visual_frame_buffer_mutex_.unlock();
        // 确保当前节点与最新视觉节点大于10ms
        if ((frame->stamp() - current_time) > 0.01) {
            break;
        }

        usleep(WAITING_DELAY_IN_US);
    }

    // 同步到当前时刻
    while (!is_finished_) {
        visual_frame_buffer_mutex_.lock();
        frame = visual_frame_buffer_.front();
        visual_frame_buffer_mutex_.unlock();
        if ((frame->stamp() - current_time) > 0.01) {
            break;
        }
        visual_frame_buffer_.pop();
    }

    return true;
}

void VINS::writeNavResult(const IntegrationState &state) {

    // 保存结果
    vector<double> result;

    double time = state.time;
    Vector3d bg = state.bg * R2D * 3600;
    Vector3d ba = state.ba * 1e5;

    {
        // imu error file
        result.clear();

        result.push_back(time);
        result.push_back(bg[0]);
        result.push_back(bg[1]);
        result.push_back(bg[2]);
        result.push_back(ba[0]);
        result.push_back(ba[1]);
        result.push_back(ba[2]);

        result.push_back(state.sodo);
        imu_err_filesaver_->dump(result);
        imu_err_filesaver_->flush();
    }

    {
        // trajectory
        result.clear();

        result.push_back(time);
        result.push_back(state.p[0]);
        result.push_back(state.p[1]);
        result.push_back(state.p[2]);
        result.push_back(state.q.x());
        result.push_back(state.q.y());
        result.push_back(state.q.z());
        result.push_back(state.q.w());
        traj_filesaver_->dump(result);
        traj_filesaver_->flush();
    }
}
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

#ifndef VINS_H
#define VINS_H

#include "fileio/filesaver.h"
#include "le_vins/optimizer.h"
#include "visual/tracking.h"
#include "visual/visual_drawer.h"

#include "preintegration/preintegration.h"

#include <ceres/ceres.h>
#include <tbb/tbb.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <unordered_map>

using visual::VisualFrame;

class VINS {

public:
    enum VINSState {
        VINS_ERROR               = -1,
        VINS_INITIALIZING        = 0,
        VINS_INITIALIZING_VISUAL = 1,
        VINS_NORMAL              = 2,
    };

    enum NodeType {
        NODE_IMU    = 0,
        NODE_VISUAL = 1,
    };

    typedef std::shared_ptr<VINS> Ptr;
    typedef std::unique_lock<std::mutex> Lock;

    VINS() = delete;
    explicit VINS(const string &configfile, const string &outputpath, visual::VisualDrawer::Ptr visual_drawer);

    bool addNewImu(const IMU &imu);
    bool addNewVisualFrame(const VisualFrame::Ptr &frame);
    bool addNewPointCloud(PointCloudCustomPtr pointcloud);
    bool canAddData();
    bool isBufferEmpty();

    void setFinished();

    bool isRunning() const {
        return !is_finished_;
    }

    VINSState vinsState() const {
        return vinsstate_;
    }

private:
    // 数据处理
    bool vinsInitialization(double last_time, double current_time);
    int vinsVisualTrackingProcessing(VisualFrame::Ptr frame);
    bool vinsOptimizationProcessing();
    bool vinsVisualVisualization();

    // 时间节点
    IntegrationState addNewTimeNode(double time, NodeType type);
    void addNewVisualFrameTimeNode(VisualFrame::Ptr frame);
    bool removeUnusedTimeNode();

    // INS数据
    bool waitImuData(double time);
    void doInsMechanization(double time);
    bool waitImuAddToInsWindow(double time);
    bool waitImuDoInsMechanization(double time);

    // 优化参数和因子
    void updateParametersFromOptimizer();
    bool removeNonKeyFrame();
    void parametersStatistic();

    // 线程
    void runFusion();

    void writeNavResult(const IntegrationState &state);

    void accumulateStationaryPointcloud(const IntegrationState &state, double end_time);
    bool accumulateDynamicPointcloud(double end_time);
    bool pointcloudStackForVisualTracking(const VisualFrame::Ptr frame, PointCloud &pointcloud);

private:
    // 正常重力
    const double NORMAL_GRAVITY = 9.8;

    // INS窗口内的最大数量, 对于200Hz, 保留5秒数据
    const size_t MAXIMUM_INS_NUMBER = 1000;
    const size_t RESERVED_INS_NUM   = 400; // INS窗口内保留的数量

    const double RESERVED_BUFFER_LENGTH = 10.0; // Buffer保存的数据时间长度
    const uint32_t WAITING_DELAY_IN_US  = 100;  // 数据等待微秒时间

    // 动态航向初始的最小速度
    const double MINMUM_ALIGN_VELOCITY = 0.5; // 0.5 m/s
    // INS初始化节点的时间间隔
    const double INS_INITIALIZATION_INTERVAL = 1.0; // 1.0 s
    // 允许的最长预积分时间
    const double MAXIMUM_PREINTEGRATION_LENGTH = 1.0; // 1.0 s

    // 激光深度增强点云累积时间
    const double LIDAR_DEPTH_POINT_STACK_INTERVAL = 0.5; // 0.5 s

    // 时间节点类型
    const std::map<NodeType, string> NODE_NAME = {{NODE_IMU, "IMU"}, {NODE_VISUAL, "visual"}};

    // 优化参数, 使用deque容器管理, 移除头尾不会造成数据内存移动
    deque<PreBasePtr> preintegrationlist_;
    deque<StateData> statedatalist_;
    deque<double> timelist_;

    double unused_time_node_{0};

    // 融合对象
    visual::Tracking::Ptr tracking_;
    visual::VisualMap::Ptr visual_map_;
    visual::Camera::Ptr camera_;
    visual::VisualDrawer::Ptr visual_drawer_;
    Optimizer::Ptr optimizer_;

    // 多线程
    tbb::task_group task_group_;
    std::thread visual_drawer_thread_;
    std::thread fusion_thread_;

    std::atomic<bool> is_finished_{false};
    std::atomic<double> start_time_{0};

    std::mutex imu_buffer_mutex_;
    std::mutex visual_frame_buffer_mutex_;
    std::mutex lidar_point_buffer_mutex_;
    std::mutex write_state_mutex_;

    // 传感器数据
    std::queue<VisualFrame::Ptr> visual_frame_buffer_;
    std::queue<IMU> imu_buffer_;
    std::deque<std::pair<IMU, IntegrationState>> ins_window_;
    std::deque<PointTypeCustom> lidar_point_buffer_;
    vector<IntegrationState> write_state_buffer_;

    // IMU参数
    std::shared_ptr<IntegrationParameters> integration_parameters_;
    PreintegrationOptions preintegration_options_;
    IntegrationConfiguration integration_config_;

    double imudatarate_{200};
    double imudatadt_{0.005};

    size_t reserved_buffer_counts_;

    // 传感器使用
    bool is_use_visual_{true};
    bool is_use_lidar_depth_{false};

    // 外参
    Pose pose_b_c_, pose_c_l_;
    double td_b_c_, td_b_l_;

    bool is_use_visualization_{true};

    // 优化选项
    bool optimize_estimate_cam_extrinsic_;
    bool optimize_estimate_cam_td_;
    bool optimize_calculate_covariance_;

    // 文件IO
    FileSaver::Ptr imu_err_filesaver_;
    FileSaver::Ptr visual_pts_filesaver_;
    FileSaver::Ptr visual_stat_filesaver_;
    FileSaver::Ptr cam_ext_filesaver_;
    FileSaver::Ptr traj_filesaver_;

    // 统计参数
    double stat_visual_tracking_cost_;

    // 系统状态
    std::atomic<VINSState> vinsstate_{VINS_ERROR};
};

#endif // VINS_VINS_H

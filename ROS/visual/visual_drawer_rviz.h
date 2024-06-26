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

#ifndef VISUAL_DRAWER_RVIZ_H
#define VISUAL_DRAWER_RVIZ_H

#include "visual/visual_drawer.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

#include <nav_msgs/Path.h>
#include <ros/ros.h>

using std::string;
using std::vector;

class VisualDrawerRviz : public visual::VisualDrawer {

public:
    explicit VisualDrawerRviz(ros::NodeHandle &nh);

    void run() override;

    void setFinished() override;

    // 地图
    void updateNewFixedMappoints(vector<Vector3d> points) override;
    void updateCurrentMappoints(vector<Vector3d> points) override;
    void updateMap(const Pose &pose) override;

    // 跟踪图像
    void updateFrame(visual::VisualFrame::Ptr frame) override;
    void updateTrackedMapPoints(vector<cv::Point2f> map, vector<cv::Point2f> matched,
                                vector<visual::MapPointType> mappoint_type) override;
    void updateTrackedRefPoints(vector<cv::Point2f> ref, vector<cv::Point2f> cur) override;

private:
    void publishTrackingImage();

    void publishMapPoints();

    void publishOdometry();

private:
    // 多线程
    std::condition_variable update_sem_;
    std::mutex update_mutex_;
    std::mutex map_mutex_;
    std::mutex image_mutex_;

    // 标志
    std::atomic<bool> is_finished_;
    std::atomic<bool> isframerdy_;
    std::atomic<bool> ismaprdy_;

    // 跟踪
    visual::VisualFrame::Ptr frame_;
    cv::Mat raw_image_;
    cv::Mat track_image_;
    vector<Vector3d> fixed_mappoints_;
    vector<Vector3d> current_mappoints_;

    Pose pose_;
    nav_msgs::Path path_;

    ros::Publisher path_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher track_image_pub_;
    ros::Publisher raw_image_pub_;
    ros::Publisher fixed_points_pub_;
    ros::Publisher current_points_pub_;

    string frame_id_;
};

#endif // VISUAL_DRAWER_RVIZ_H

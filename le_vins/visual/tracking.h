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

#ifndef VISUAL_TRACKING_H
#define VISUAL_TRACKING_H

#include "lidar/lidar.h"
#include "visual/camera.h"
#include "visual/visual_drawer.h"
#include "visual/visual_feature.h"
#include "visual/visual_frame.h"
#include "visual/visual_map.h"
#include "visual/visual_mappoint.h"

#include <pcl/kdtree/kdtree_flann.h>

namespace visual {

using std::string;
using std::vector;

typedef enum TrackState {
    TRACK_FIRST_FRAME = 0,
    TRACK_INITIALIZING,
    TRACK_TRACKING,
    TRACK_PASSED,
    TRACK_LOST,
} TrackState;

class Tracking {

public:
    typedef std::shared_ptr<Tracking> Ptr;

    Tracking(Camera::Ptr camera, VisualMap::Ptr map, VisualDrawer::Ptr drawer, const string &configfile,
             const string &outputpath);

    TrackState track(VisualFrame::Ptr frame, PointCloud &pointcloud);

    bool isNewKeyFrame() const {
        return is_new_keyframe_;
    }

    double trackingTimeCost() const {
        return tracking_timecost_;
    }

private:
    bool preprocessing(VisualFrame::Ptr frame);

    bool doResetTracking(bool is_reserved);
    bool trackReferenceFrame();
    bool trackMappoint();
    void showTracking();

    void featuresDetection(bool ismask);
    bool triangulation();
    static void triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                 const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw);

    double relativeTranslation();
    double relativeRotation();

    void makeNewFrame(int state);
    keyFrameState checkKeyFrameSate();

    double keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0, const Pose &pose1);

    bool isOnBorder(const cv::Point2f &pts);
    static double ptsDistance(cv::Point2f &pt1, cv::Point2f &pt2);

    template <typename T> static void reduceVector(T &vec, vector<uint8_t> status);
    static Eigen::Matrix4d pose2Tcw(const Pose &pose);

    bool depthAssociationProjection(PointCloud &pointcloud);
    int depthAssociationReference();
    int depthAssociationMapPoint();
    MapPointType depthAssociationPlaneFitting(PointType &point);

public:
    static constexpr double ASSOCIATE_DEPTH_STD = 0.1;

private:
    // 配置参数
    const double TRACK_BLOCK_SIZE   = 200.0; // 特征提取分块大小
    const int TRACK_MAX_ITERATION   = 30;    // 光流跟踪最大迭代次数
    const double TRACK_MIN_PARALLAX = 10.0;  // 三角化时最小的像素视差
    const double TRACK_MIN_INTERVAl = 0.096; // 最短观测帧时间间隔

    // 关键帧选择最小旋转角度
    const double TRACK_MIN_ROTATION = 10 * D2R; // 10 deg
    // 关键帧选择最小平移距离
    const double TRACK_MIN_TRANSLATION = 0.2; // 0.2 m
    // 关键帧选择最长时间间隔
    const double TRACK_MAX_INTERVAl = 0.5 * 0.95; // 0.5 s
    // 关键帧选择最小时间间隔
    const double TRACK_DEFAULT_INTERVAl = 0.3 * 0.95; // 0.3 s

    // 激光点云深度关联参数

    // 距离图像的角度分辨率
    const double ASSOCIATE_RANGE_IMAGE_ANGLE = 0.3 * M_PI / 180.0;
    // 最近邻点的最大平方距离为最小间隔的三倍
    const int ASSOCIATE_NEAREST_DISTANCE_COUNTS = 3;

    // 相机视场角
    const double ASSOCIATE_IMAGE_FOV = 90.0 * M_PI / 180.0;
    // 距离图像的大小
    const int ASSOCIATE_IMAGE_SIZE = static_cast<int>(ASSOCIATE_IMAGE_FOV / ASSOCIATE_RANGE_IMAGE_ANGLE);
    // 图像坐标系下的角度正切值
    const double ASSOCIATE_POINT_TAN_VALUE = tan(ASSOCIATE_IMAGE_FOV * 0.5);
    // 最近邻点的最大距离
    const double ASSOCIATE_NEAREST_DISTANCE_THRESHOLD =
        sin(ASSOCIATE_RANGE_IMAGE_ANGLE) * ASSOCIATE_NEAREST_DISTANCE_COUNTS;
    // 最近邻点的最大距离
    const double ASSOCIATE_NEAREST_SQUARE_DISTANCE_THRESHOLD = pow(ASSOCIATE_NEAREST_DISTANCE_THRESHOLD, 2);
    // 单个距离图像最小的平方距离
    const double ASSOCIATE_NEAREST_SQUARE_DISTANCE_UNIT = pow(sin(ASSOCIATE_RANGE_IMAGE_ANGLE), 2);

    // 图像帧
    VisualFrame::Ptr frame_cur_, frame_ref_, frame_pre_, last_keyframe_;

    // 相机
    Camera::Ptr camera_;

    // 地图
    VisualMap::Ptr visual_map_;

    // 绘图
    VisualDrawer::Ptr visual_drawer_;

    // 图像处理
    cv::Ptr<cv::CLAHE> clahe_;

    // 特征点
    vector<cv::Point2f> pts2d_cur_, pts2d_new_, pts2d_ref_;
    vector<VisualFrame::Ptr> pts2d_ref_frame_;
    vector<Eigen::Vector2d> velocity_ref_, velocity_cur_;
    vector<VisualMapPoint::Ptr> tracked_mappoint_, mappoint_matched_;

    // 激光深度辅助
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_;
    PointCloudPtr unit_sphere_pointcloud_;

    // 分块特征提取, 第一个为分块的长宽, 然后是按照一行一行的分块起始坐标
    vector<std::pair<int, int>> block_indexs_;
    int block_cols_, block_rows_, block_cnts_;
    int track_max_block_features_;

    bool is_new_keyframe_{false};
    bool is_initializing_{true};

    // configurations
    int track_max_features_;
    int track_min_pixel_distance_;
    double reprojection_error_std_;
    bool is_use_visualization_;

    double tracking_timecost_{0};
};

} // namespace visual

#endif // VISUAL_TRACKING_H

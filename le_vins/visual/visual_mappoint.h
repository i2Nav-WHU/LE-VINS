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

#ifndef VISUAL_MAPPOINT_H
#define VISUAL_MAPPOINT_H

#include "visual/camera.h"
#include "visual/visual_feature.h"

#include <Eigen/Geometry>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace visual {

using cv::Mat;
using Eigen::Vector3d;

enum MapPointType {
    MAPPOINT_NONE             = -1,
    MAPPOINT_TRIANGULATED     = 0,
    MAPPOINT_DEPTH_ASSOCIATED = 1,
};

class VisualMapPoint {

public:
    typedef std::shared_ptr<VisualMapPoint> Ptr;

    static constexpr double DEFAULT_DEPTH  = 10.0;
    static constexpr double NEAREST_DEPTH  = 1;   // 最近可用路标点深度
    static constexpr double FARTHEST_DEPTH = 200; // 最远可用路标点深度

    VisualMapPoint() = delete;
    VisualMapPoint(ulong id, const std::shared_ptr<VisualFrame> &ref_frame, Vector3d pos, cv::Point2f keypoint,
                   double depth, MapPointType type);

    static VisualMapPoint::Ptr createMapPoint(std::shared_ptr<VisualFrame> &ref_frame, Vector3d &pos,
                                              cv::Point2f &keypoint, double depth, MapPointType type);

    static bool isGoodToTrack(const Camera::Ptr &camera, const cv::Point2f &pp, const Pose &pose, const Vector3d &pw,
                              double reproj_std, double depth_scale);

    Vector3d &pos() {

        return pos_;
    };

    int observedTimes() const {
        return observed_times_;
    }

    ulong id() const {
        return id_;
    }

    void addObservation(const VisualFeature::Ptr &feature);

    void increaseUsedTimes() {
        used_times_++;
    }

    void decreaseUsedTimes() {
        if (used_times_) {
            used_times_--;
        }
    }

    int usedTimes() {
        return used_times_;
    }

    void addOptimizedTimes() {
        optimized_times_++;
    }

    int optimizedTimes() {
        return optimized_times_;
    }

    void removeAllObservations() {
        observations_.clear();
    }

    const std::vector<std::weak_ptr<VisualFeature>> &observations() {
        return observations_;
    }

    const std::vector<cv::Point2f> featurePoints() {
        return feature_points_;
    }

    void setOutlier(bool isoutlier) {
        isoutlier_ = isoutlier;
    }

    bool isOutlier() {
        return isoutlier_;
    }

    double depth() {
        return depth_;
    }

    void updateDepth(double depth) {
        depth_ = depth;
    }

    MapPointType &mapPointType() {
        return mappoint_type_;
    }

    std::shared_ptr<VisualFrame> referenceFrame() {
        return ref_frame_.lock();
    }

    std::shared_ptr<VisualFrame> farthestKeyFrame() {
        return farthest_keyframe_.lock();
    }

    void setFarthestFrame(std::shared_ptr<VisualFrame> frame) {
        farthest_keyframe_ = frame;
    }

    const cv::Point2f &referenceKeypoint() {
        return ref_frame_keypoint_;
    }

    double lidarDepth() {
        return lidar_depth_;
    }

    void setReferenceFrame(const std::shared_ptr<VisualFrame> &frame, Vector3d pos, cv::Point2f keypoint, double depth,
                           MapPointType type);
    static bool isGoodDepth(double depth, double scale);

private:
    std::vector<std::weak_ptr<VisualFeature>> observations_;
    std::vector<cv::Point2f> feature_points_;

    Vector3d pos_;

    // 参考帧中的深度
    double depth_{DEFAULT_DEPTH}, lidar_depth_{DEFAULT_DEPTH};
    cv::Point2f ref_frame_keypoint_;
    std::weak_ptr<VisualFrame> ref_frame_, farthest_keyframe_;

    int optimized_times_;
    int used_times_;
    int observed_times_;
    bool isoutlier_;

    ulong id_;
    MapPointType mappoint_type_{MAPPOINT_NONE};
};

} // namespace visual

#endif // VISUAL_MAPPOINT_H

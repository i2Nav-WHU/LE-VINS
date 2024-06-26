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

#include "visual/visual_mappoint.h"
#include "visual/visual_frame.h"

namespace visual {

VisualMapPoint::VisualMapPoint(ulong id, const std::shared_ptr<VisualFrame> &ref_frame, Vector3d pos,
                               cv::Point2f keypoint, double depth, MapPointType type)
    : pos_(std::move(pos))
    , depth_(depth)
    , ref_frame_keypoint_(std::move(keypoint))
    , ref_frame_(ref_frame)
    , optimized_times_(0)
    , used_times_(0)
    , observed_times_(0)
    , isoutlier_(false)
    , id_(id)
    , mappoint_type_(type) {

    // 防止深度错误
    if ((depth_ < NEAREST_DEPTH) || (depth_ > FARTHEST_DEPTH)) {
        depth_ = DEFAULT_DEPTH;
    }
    lidar_depth_ = depth_;
}

VisualMapPoint::Ptr VisualMapPoint::createMapPoint(std::shared_ptr<VisualFrame> &ref_frame, Vector3d &pos,
                                                   cv::Point2f &feature, double depth, MapPointType type) {
    static ulong factory_id_ = 0;
    return std::make_shared<VisualMapPoint>(factory_id_++, ref_frame, pos, feature, depth, type);
}

void VisualMapPoint::addObservation(const VisualFeature::Ptr &feature) {
    observations_.push_back(feature);
    feature_points_.push_back(feature->keyPoint());
    observed_times_++;
}

void VisualMapPoint::setReferenceFrame(const std::shared_ptr<VisualFrame> &frame, Vector3d pos, cv::Point2f keypoint,
                                       double depth, MapPointType type) {
    depth_ = depth;
    if (depth_ < 1.0) {
        depth_ = DEFAULT_DEPTH;
    }
    lidar_depth_ = depth_;

    pos_                = std::move(pos);
    ref_frame_          = frame;
    ref_frame_keypoint_ = std::move(keypoint);
    mappoint_type_      = type;
}

bool VisualMapPoint::isGoodToTrack(const Camera::Ptr &camera, const cv::Point2f &pp, const Pose &pose,
                                   const Vector3d &pw, double reproj_std, double depth_scale) {
    // 当前相机坐标系
    Vector3d pc = camera->world2cam(pw, pose);

    // 深度检查
    if (!isGoodDepth(pc[2], depth_scale)) {
        return false;
    }

    // 重投影误差检查
    if (camera->reprojectionError(pose, pw, pp).norm() > reproj_std) {
        return false;
    }

    return true;
}

bool VisualMapPoint::isGoodDepth(double depth, double scale) {
    return ((depth > NEAREST_DEPTH) && (depth < FARTHEST_DEPTH * scale));
}

} // namespace visual

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

#ifndef VISUAL_FEATURE_H
#define VISUAL_FEATURE_H

#include <memory>
#include <opencv2/opencv.hpp>

namespace visual {

using cv::Mat;

class VisualFrame;
class VisualMapPoint;

class VisualFeature {

public:
    typedef std::shared_ptr<VisualFeature> Ptr;

    enum FeatureType {
        FEATURE_NONE             = -1,
        FEATURE_MATCHED          = 0,
        FEATURE_TRIANGULATED     = 1,
        FEATURE_DEPTH_ASSOCIATED = 2,
    };

    VisualFeature() = delete;
    VisualFeature(const std::shared_ptr<VisualFrame> &frame, const Eigen::Vector2d &velocity, cv::Point2f keypoint,
                  cv::Point2f distorted, FeatureType type)
        : frame_(frame)
        , keypoint_(std::move(keypoint))
        , distorted_keypoint_(std::move(distorted))
        , isoutlier_(false)
        , type_(type) {

        velocity_[0] = velocity[0];
        velocity_[1] = velocity[1];
        velocity_[2] = 0;
    }

    static std::shared_ptr<VisualFeature> createFeature(const std::shared_ptr<VisualFrame> &frame,
                                                        const Eigen::Vector2d &velocity, const cv::Point2f &keypoint,
                                                        const cv::Point2f &distorted, FeatureType type) {
        return std::make_shared<VisualFeature>(frame, velocity, keypoint, distorted, type);
    }

    std::shared_ptr<VisualFrame> getFrame() {
        return frame_.lock();
    }

    std::shared_ptr<VisualMapPoint> getMapPoint() {
        return mappoint_.lock();
    }

    const cv::Point2f &keyPoint() {
        return keypoint_;
    }

    const cv::Point2f &distortedKeyPoint() {
        return distorted_keypoint_;
    }

    void addMapPoint(const std::shared_ptr<VisualMapPoint> &mappoint) {
        mappoint_ = mappoint;
    }

    void setOutlier(bool isoutlier) {
        isoutlier_ = isoutlier;
    }

    bool isOutlier() const {
        return isoutlier_;
    }

    FeatureType featureType() {
        return type_;
    }

    const Vector3d &velocityInPixel() {
        return velocity_;
    }

    void setVelocityInPixel(const cv::Point2f &velocity) {
        velocity_[0] = velocity.x;
        velocity_[1] = velocity.y;
        velocity_[2] = 0;
    }

private:
    std::weak_ptr<VisualFrame> frame_;
    std::weak_ptr<VisualMapPoint> mappoint_;

    cv::Point2f keypoint_, distorted_keypoint_;
    Vector3d velocity_{0, 0, 0};

    bool isoutlier_;

    FeatureType type_;
};

} // namespace visual

#endif // VISUAL_FEATURE_H

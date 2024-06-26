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

#ifndef VISUAL_FRAME_H
#define VISUAL_FRAME_H

#include "common/types.h"
#include "visual/visual_feature.h"

#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace visual {

using cv::Mat;
using std::vector;

enum keyFrameState {
    KEYFRAME_NONE              = 0,
    KEYFRAME_REMOVE_SECOND_NEW = 1,
    KEYFRAME_NORMAL            = 2,
    KEYFRAME_REMOVE_OLDEST     = 3,
};

class VisualFrame {

public:
    typedef std::shared_ptr<VisualFrame> Ptr;

    VisualFrame() = delete;
    VisualFrame(ulong id, double stamp, Mat image);

    static VisualFrame::Ptr createFrame(double stamp, const Mat &image);

    void setKeyFrame(int state);

    void resetKeyFrame() {
        iskeyframe_     = false;
        keyframe_state_ = KEYFRAME_NONE;
    }

    const Mat &image() {
        return image_;
    }

    const Mat &rawImage() {
        return raw_image_;
    }

    const Pose &pose() {
        return pose_;
    }

    void setPose(Pose pose) {
        pose_ = std::move(pose);
    }

    const std::unordered_map<ulong, VisualFeature::Ptr> &features() {
        return features_;
    }

    void clearFeatures() {
        features_.clear();
        unupdated_mappoints_.clear();
    }

    size_t numFeatures() {
        return features_.size();
    }

    const std::vector<std::shared_ptr<VisualMapPoint>> &unupdatedMappoints() {
        return unupdated_mappoints_;
    }

    void addNewUnupdatedMappoint(const std::shared_ptr<VisualMapPoint> &mappoint) {
        unupdated_mappoints_.push_back(mappoint);
    }

    void addFeature(ulong mappointid, const VisualFeature::Ptr &features) {
        features_.insert(make_pair(mappointid, features));
    }

    double stamp() const {
        return stamp_;
    }

    void setStamp(double stamp) {
        stamp_ = stamp;
    }

    double timeDelay() const {
        return td_;
    }

    double &timeDelayEst() {
        return td_est_;
    }

    void setTimeDelay(double td) {
        td_     = td;
        td_est_ = td;
    }

    bool isKeyFrame() const {
        return iskeyframe_;
    }

    ulong id() const {
        return id_;
    }

    ulong keyFrameId() const {
        return keyframe_id_;
    }

    void setKeyFrameState(int state) {
        keyframe_state_ = state;
    }

    int keyFrameState() {
        return keyframe_state_;
    }

private:
    int keyframe_state_{KEYFRAME_NORMAL};

    ulong id_;
    ulong keyframe_id_;

    double stamp_;
    double td_{0}, td_est_{0};

    // 世界坐标系下的位姿
    Pose pose_;

    Mat image_, raw_image_;

    bool iskeyframe_;

    std::unordered_map<ulong, VisualFeature::Ptr> features_;
    vector<std::shared_ptr<VisualMapPoint>> unupdated_mappoints_;
};

} // namespace visual

#endif // VISUAL_FRAME_H

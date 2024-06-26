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

#ifndef VISUAL_MAP_H
#define VISUAL_MAP_H

#include "visual/visual_frame.h"
#include "visual/visual_mappoint.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace visual {

class VisualMap {

public:
    typedef std::shared_ptr<VisualMap> Ptr;

    typedef std::unordered_map<ulong, VisualFrame::Ptr> KeyFrames;
    typedef std::unordered_map<ulong, VisualMapPoint::Ptr> LandMarks;

    VisualMap() = delete;
    explicit VisualMap(size_t size)
        : window_size_(size) {
        keyframes_.clear();
        landmarks_.clear();
    }

    void resetWindowSize(size_t size) {
        window_size_ = size;
    }

    size_t windowSize() const {
        return window_size_;
    }

    void insertKeyFrame(const VisualFrame::Ptr &frame);

    const KeyFrames &keyframes() {
        return keyframes_;
    }

    const LandMarks &landmarks() {
        return landmarks_;
    }

    vector<ulong> orderedKeyFrames();
    VisualFrame::Ptr oldestKeyFrame();
    const VisualFrame::Ptr &latestKeyFrame();

    void removeMappoint(VisualMapPoint::Ptr &mappoint);
    void removeKeyFrame(VisualFrame::Ptr &frame, bool is_remove_mappoint);

    double mappointObservedRate(const VisualMapPoint::Ptr &mappoint);

    bool isWindowFull() {
        return keyframes_.size() > window_size_;
    }

    bool isWindowNormal() {
        return keyframes_.size() == window_size_;
    }

    bool isWindowHalfFull() {
        return keyframes_.size() > window_size_ / 2;
    }

    bool isKeyFrameInMap(const VisualFrame::Ptr &frame) {
        return keyframes_.find(frame->keyFrameId()) != keyframes_.end();
    }

private:
    // 局部地图
    KeyFrames keyframes_;
    LandMarks landmarks_;

    VisualFrame::Ptr latest_keyframe_;

    size_t window_size_{10};
};

} // namespace visual

#endif // VISUAL_MAP_H

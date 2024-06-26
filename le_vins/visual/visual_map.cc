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

#include "visual/visual_map.h"
#include "visual/visual_frame.h"
#include "visual/visual_mappoint.h"

namespace visual {

void VisualMap::insertKeyFrame(const VisualFrame::Ptr &frame) {
    // New keyframe
    latest_keyframe_ = frame;
    if (keyframes_.find(frame->keyFrameId()) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyFrameId(), frame));
    } else {
        keyframes_[frame->keyFrameId()] = frame;
    }

    // New mappoints
    auto &unupdated_mappoints = frame->unupdatedMappoints();
    for (const auto &mappoint : unupdated_mappoints) {
        if (landmarks_.find(mappoint->id()) == landmarks_.end()) {
            landmarks_.insert(make_pair(mappoint->id(), mappoint));
        } else {
            landmarks_[mappoint->id()] = mappoint;
        }
    }
}

vector<ulong> VisualMap::orderedKeyFrames() {
    vector<ulong> keyframeid;
    for (auto &keyframe : keyframes_) {
        keyframeid.push_back(keyframe.first);
    }
    std::sort(keyframeid.begin(), keyframeid.end());

    return keyframeid;
}

VisualFrame::Ptr VisualMap::oldestKeyFrame() {
    auto oldest = orderedKeyFrames()[0];
    return keyframes_.at(oldest);
}

const VisualFrame::Ptr &VisualMap::latestKeyFrame() {
    return latest_keyframe_;
}

void VisualMap::removeMappoint(VisualMapPoint::Ptr &mappoint) {
    mappoint->setOutlier(true);
    mappoint->removeAllObservations();
    if (landmarks_.find(mappoint->id()) != landmarks_.end()) {
        landmarks_.erase(mappoint->id());
    }
    mappoint.reset();
}

void VisualMap::removeKeyFrame(VisualFrame::Ptr &frame, bool is_remove_mappoint) {
    if (is_remove_mappoint) {
        // 移除与关键帧关联的所有路标点
        vector<ulong> mappointid;
        const auto &features = frame->features();
        for (const auto &feature : features) {
            const auto &mappoint = feature.second->getMapPoint();
            if (mappoint) {
                // 参考帧非边缘化帧, 不移除
                auto ref_frame = mappoint->referenceFrame();
                if (ref_frame && (ref_frame != frame)) {
                    bool is_in_map = keyframes_.find(ref_frame->keyFrameId()) != keyframes_.end();
                    if (is_in_map) {
                        continue;
                    }
                }
                mappointid.push_back(mappoint->id());
            }
        }
        for (auto id : mappointid) {
            const auto &landmark = landmarks_.find(id);
            if (landmark != landmarks_.end()) {
                auto mappoint = landmark->second;
                if (mappoint) {
                    mappoint->removeAllObservations();
                    // 强制设置为outlier
                    mappoint->setOutlier(true);
                }
                landmarks_.erase(id);
            }
        }
        frame->clearFeatures();
    }

    // 移除关键帧
    keyframes_.erase(frame->keyFrameId());
}

double VisualMap::mappointObservedRate(const VisualMapPoint::Ptr &mappoint) {
    size_t num_keyframes = keyframes_.size();
    size_t num_observed  = 0;

    for (auto &feature : mappoint->observations()) {
        auto feat = feature.lock();
        if (!feat) {
            continue;
        }
        auto frame = feat->getFrame();
        if (!frame) {
            continue;
        }

        if (keyframes_.find(frame->keyFrameId()) != keyframes_.end()) {
            num_observed += 1;
        }
    }
    return static_cast<double>(num_observed) / static_cast<double>(num_keyframes);
}

} // namespace visual

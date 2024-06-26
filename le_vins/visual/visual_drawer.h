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

#ifndef VISUAL_DRAWER_H
#define VISUAL_DRAWER_H

#include "visual/visual_frame.h"
#include "visual/visual_map.h"

#include <memory>

namespace visual {

class VisualDrawer {

public:
    typedef std::shared_ptr<VisualDrawer> Ptr;

    virtual ~VisualDrawer() = default;

    virtual void run()         = 0;
    virtual void setFinished() = 0;

    // 地图
    virtual void updateNewFixedMappoints(vector<Vector3d> points) = 0;
    virtual void updateCurrentMappoints(vector<Vector3d> points)  = 0;
    virtual void updateMap(const Pose &pose)                      = 0;

    // 跟踪图像
    virtual void updateFrame(VisualFrame::Ptr frame)                                        = 0;
    virtual void updateTrackedMapPoints(vector<cv::Point2f> map, vector<cv::Point2f> matched,
                                        vector<MapPointType> mappoint_type) = 0;
    virtual void updateTrackedRefPoints(vector<cv::Point2f> ref, vector<cv::Point2f> cur)   = 0;

protected:
    void drawTrackingImage(const Mat &raw, Mat &drawed);

protected:
    vector<cv::Point2f> pts2d_cur_, pts2d_ref_, pts2d_map_, pts2d_matched_;
    vector<MapPointType> mappoint_type_;
};

} // namespace visual

#endif // VISUAL_DRAWER_H

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

#include "lidar_converter.h"

#include "common/gpstime.h"
#include "common/logging.h"

#include <pcl_conversions/pcl_conversions.h>

LidarConverter::LidarConverter(double frame_rate, int scan_line, double nearest_distance,
                               double farthest_distance)
    : scan_line_(scan_line)
    , frame_rate_(frame_rate) {
    nearest_distance_         = nearest_distance;
    nearest_square_distance_  = nearest_distance * nearest_distance;
    farthest_square_distance_ = farthest_distance * farthest_distance;
}

size_t LidarConverter::livoxPointCloudConvertion(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                 PointCloudCustomPtr &pointcloud_raw,
                                                 PointCloudCustomPtr &pointcloud_ds, double &start, double &end,
                                                 bool to_gps_time) {
    pointcloud_raw->clear();
    pointcloud_raw->reserve(msg->points.size());
    pointcloud_ds->clear();
    pointcloud_ds->reserve(msg->points.size());

    // 使用统一的时间
    uint64_t timebase = msg->header.stamp.toNSec();

    int num_points = 0;
    int filter_num = point_filter_num_;

    start = DBL_MAX;
    end   = 0;
    for (size_t k = 0; k < msg->points.size(); k++) {
        const auto &raw = msg->points[k];
        // 第0回波或第1回波
        if ((raw.line < scan_line_) && ((raw.tag & 0x30) == 0x00 || (raw.tag & 0x30) == 0x10)) {
            PointTypeCustom point = livoxPointConvertion(raw, timebase, to_gps_time);

            if (point.time < start) {
                start = point.time;
            }
            if (point.time > end) {
                end = point.time;
            }

            // 有效测量距离范围, 无效点数值为0
            double square_dist = point.getVector3fMap().squaredNorm();
            if ((square_dist > nearest_square_distance_) && (square_dist < farthest_square_distance_)) {
                // 原始点
                pointcloud_raw->push_back(point);

                // 抽取滤波降采样
                num_points++;
                if (num_points % filter_num == 0) {
                    pointcloud_ds->push_back(point);
                }
            }
        }
    }

    return pointcloud_ds->size();
}

size_t LidarConverter::velodynePointCloudConvertion(const sensor_msgs::PointCloud2ConstPtr &msg,
                                                    PointCloudCustomPtr &pointcloud, double &start, double &end,
                                                    bool to_gps_time) {
    pointcloud->clear();

    pcl::PointCloud<VelodynePoint> pointcloud_raw;
    pcl::fromROSMsg(*msg, pointcloud_raw);

    // 数据包时间
    double stamp = msg->header.stamp.toSec();
    if (to_gps_time) {
        int week;
        double sow;
        GpsTime::unix2gps(stamp, week, sow);
        stamp = sow;
    }

    // 根据最后一个点判断是否需要计算时间偏差
    bool is_need_time = false;
    if (pointcloud_raw.back().time == 0) {
        is_need_time = true;
        LOGW << "Lidar frame without time stamp";
    }

    std::vector<int> point_counts(scan_line_, 0);

    // 雷达扫描角速率 rad/s
    static const double omega_rate = 2 * M_PI * frame_rate_;
    // 第一个点的航向角
    double first_yaw = 0;
    for (size_t k = 0; k < pointcloud_raw.size(); k++) {
        if (!pointcloud_raw[k].getVector3fMap().isZero()) {
            first_yaw = atan2(pointcloud_raw[k].y, pointcloud_raw[k].x);
            break;
        }
    }

    PointTypeCustom point;

    double offset_time;
    start = DBL_MAX;
    end   = 0;
    for (size_t k = 0; k < pointcloud_raw.size(); k++) {
        auto &raw = pointcloud_raw[k];
        // 按照扫描线降采样
        if (point_counts[raw.ring] % point_filter_num_ == 0) {
            point.getVector3fMap() = raw.getVector3fMap();
            point.intensity        = raw.intensity;

            // 获取时间偏移
            if (is_need_time) {
                double yaw = atan2(point.y, point.x);

                // 计算偏移
                if (yaw >= first_yaw) {
                    offset_time = (yaw - first_yaw) / omega_rate;
                } else {
                    offset_time = (yaw - first_yaw + 2 * M_PI) / omega_rate;
                }
            } else {
                offset_time = raw.time;
            }

            point.time = stamp + offset_time;
            if (point.time < start) {
                start = point.time;
            }
            if (point.time > end) {
                end = point.time;
            }

            // 不在距离范围内
            double square_dist = point.getVector3fMap().squaredNorm();
            if ((square_dist > nearest_square_distance_) && (square_dist < farthest_square_distance_)) {
                pointcloud->push_back(point);
            }
        }
        point_counts[raw.ring]++;
    }

    return pointcloud->size();
}

size_t LidarConverter::ousterPointCloudConvertion(const sensor_msgs::PointCloud2ConstPtr &msg,
                                                  PointCloudCustomPtr &pointcloud, double &start, double &end,
                                                  bool to_gps_time) {
    pointcloud->clear();

    pcl::PointCloud<OusterPoint> pointcloud_raw;
    pcl::fromROSMsg(*msg, pointcloud_raw);

    // 数据包时间
    double stamp = msg->header.stamp.toSec();
    if (to_gps_time) {
        int week;
        double sow;
        GpsTime::unix2gps(stamp, week, sow);
        stamp = sow;
    }

    std::vector<int> point_counts(scan_line_, 0);
    PointTypeCustom point;

    start = DBL_MAX;
    end   = 0;
    for (size_t k = 0; k < pointcloud_raw.size(); k++) {
        auto &raw = pointcloud_raw[k];
        // 按照扫描线降采样
        if (point_counts[raw.ring] % point_filter_num_ == 0) {
            point.getVector3fMap() = raw.getVector3fMap();
            point.intensity        = raw.intensity;
            point.time             = stamp + raw.t * 1.0e-9;

            if (point.time < start) {
                start = point.time;
            }
            if (point.time > end) {
                end = point.time;
            }

            // 不在距离范围内
            double square_dist = point.getVector3fMap().squaredNorm();
            if ((square_dist > nearest_square_distance_) && (square_dist < farthest_square_distance_)) {
                pointcloud->push_back(point);
            }
        }
        point_counts[raw.ring]++;
    }

    return pointcloud->size();
}

PointTypeCustom LidarConverter::livoxPointConvertion(const livox_ros_driver::CustomPoint &raw, uint64_t timebase,
                                                     bool to_gps_time) {
    PointTypeCustom point;

    point.x         = raw.x;
    point.y         = raw.y;
    point.z         = raw.z;
    point.intensity = raw.reflectivity;
    point.time      = static_cast<double>(timebase + raw.offset_time) * 1.0e-9;
    if (to_gps_time) {
        int week;
        double sow;
        GpsTime::unix2gps(point.time, week, sow);
        point.time = sow;
    }

    return point;
}

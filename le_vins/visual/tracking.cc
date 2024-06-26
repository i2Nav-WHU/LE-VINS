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

#include "visual/tracking.h"

#include "common/logging.h"
#include "common/rotation.h"
#include "common/timecost.h"

#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

namespace visual {

Tracking::Tracking(Camera::Ptr camera, VisualMap::Ptr map, VisualDrawer::Ptr drawer, const string &configfile,
                   const string &outputpath)
    : frame_cur_(nullptr)
    , frame_ref_(nullptr)
    , camera_(std::move(camera))
    , visual_map_(std::move(map))
    , visual_drawer_(std::move(drawer))
    , is_new_keyframe_(false)
    , is_initializing_(true) {

    YAML::Node config;
    std::vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    track_max_features_ = config["visual"]["track_max_features"].as<int>();

    is_use_visualization_   = config["is_use_visualization"].as<bool>();
    reprojection_error_std_ = config["optimizer"]["optimize_reprojection_std"].as<double>();

    // 直方图均衡化
    clahe_ = cv::createCLAHE(3.0, cv::Size(21, 21));

    // 分块索引
    block_cols_ = static_cast<int>(lround(camera_->width() / TRACK_BLOCK_SIZE));
    block_rows_ = static_cast<int>(lround(camera_->height() / TRACK_BLOCK_SIZE));
    block_cnts_ = block_cols_ * block_rows_;

    int col, row;
    row = camera_->height() / block_rows_;
    col = camera_->width() / block_cols_;
    block_indexs_.emplace_back(std::make_pair(col, row));
    for (int i = 0; i < block_rows_; i++) {
        for (int j = 0; j < block_cols_; j++) {
            block_indexs_.emplace_back(std::make_pair(col * j, row * i));
        }
    }

    // 每个分块提取的角点数量
    track_max_block_features_ =
        static_cast<int>(lround(static_cast<double>(track_max_features_) / static_cast<double>(block_cnts_)));

    // 每个格子的提取特征数量平方面积为格子面积的 2/3
    track_min_pixel_distance_ = static_cast<int>(round(TRACK_BLOCK_SIZE / sqrt(track_max_block_features_ * 1.5)));

    frame_ref_ = nullptr;
    frame_cur_ = nullptr;

    LOGI << "Visual tracking is constructed";
}

bool Tracking::preprocessing(VisualFrame::Ptr frame) {
    is_new_keyframe_ = false;

    frame_pre_ = frame_cur_;
    frame_cur_ = frame;

    // 直方图均衡化
    clahe_->apply(frame_cur_->image(), frame_cur_->image());

    return true;
}

TrackState Tracking::track(VisualFrame::Ptr frame, PointCloud &pointcloud) {

    TrackState track_state = TRACK_PASSED;

    // 预处理
    if (!preprocessing(frame)) {
        return track_state;
    }

    if (is_initializing_) {
        // 初始化
        if ((frame_ref_ == nullptr) || (pts2d_ref_.empty())) {
            doResetTracking(true);
            return TRACK_FIRST_FRAME;
        }

        TimeCost timecost;

        // 从参考帧跟踪过来的特征点
        trackReferenceFrame();

        tracking_timecost_ = timecost.costInMillisecond();

        // 初始化需要足够的平移
        double translation = relativeTranslation();
        double dt          = frame_cur_->stamp() - frame_ref_->stamp();
        if (!((translation > TRACK_MIN_TRANSLATION) && (dt > TRACK_DEFAULT_INTERVAl))) {
            // 显示跟踪情况
            showTracking();
            return TRACK_INITIALIZING;
        }
        LOGI << "Initialization tracking with translation " << translation;

        // 首次三角化
        triangulation();

        // 初始化失败, 重置初始化
        if (doResetTracking(false)) {
            showTracking();
            return TRACK_LOST;
        }

        // 新关键帧, 地图更新, 数据转存
        makeNewFrame(KEYFRAME_NORMAL);
        last_keyframe_ = frame_cur_;

        is_initializing_ = false;

        track_state = TRACK_TRACKING;
    } else {
        // tracking

        TimeCost timecost;

        // 跟踪上一帧中带路标点的特征, 利用预测的位姿先验
        trackMappoint();

        // 未关联路标点的新特征
        trackReferenceFrame();

        tracking_timecost_ = timecost.costInMillisecond();

        // 检查关键帧类型
        auto keyframe_state = checkKeyFrameSate();

        // 正常关键帧, 需要三角化路标点
        if ((keyframe_state == KEYFRAME_NORMAL) || (keyframe_state == KEYFRAME_REMOVE_OLDEST)) {
            // 激光深度关联
            if (!pointcloud.empty()) {
                depthAssociationProjection(pointcloud);

                if (!unit_sphere_pointcloud_->empty()) {
                    // 关联已有路标点
                    depthAssociationMapPoint();

                    // 关联未三角化的路标点
                    depthAssociationReference();
                }
            }

            // 三角化补充路标点
            triangulation();
        } else {
            // 添加新的特征
            featuresDetection(true);
        }

        // 跟踪失败, 路标点数据严重不足
        if (doResetTracking(false)) {
            return TRACK_LOST;
        }

        // 观测帧, 进行插入
        if (keyframe_state != KEYFRAME_NONE) {
            makeNewFrame(keyframe_state);
        }

        track_state = TRACK_TRACKING;
    }

    // 显示跟踪情况
    showTracking();

    return track_state;
}

void Tracking::makeNewFrame(int state) {
    // 插入关键帧
    frame_cur_->setKeyFrame(state);
    is_new_keyframe_ = true;

    // 仅当正常关键帧才更新参考帧
    if ((state == KEYFRAME_NORMAL) || (state == KEYFRAME_REMOVE_OLDEST)) {
        // 参考帧变化
        frame_ref_ = frame_cur_;

        // 添加新的特征
        featuresDetection(true);
    }
}

keyFrameState Tracking::checkKeyFrameSate() {
    keyFrameState keyframe_state = KEYFRAME_NONE;

    // 相邻时间太短, 不进行关键帧处理
    double dt = frame_cur_->stamp() - last_keyframe_->stamp();
    if (dt < TRACK_MIN_INTERVAl) {
        return keyframe_state;
    }

    double translation = relativeTranslation();
    double rotation    = relativeRotation();
    if (((translation > TRACK_MIN_TRANSLATION) && (dt > TRACK_DEFAULT_INTERVAl)) || (rotation > TRACK_MIN_ROTATION)) {
        // 新的关键帧, 满足最小像素视差

        keyframe_state = visual_map_->isWindowNormal() ? KEYFRAME_REMOVE_OLDEST : KEYFRAME_NORMAL;
        LOGI << "Keyframe at " << Logging::doubleData(frame_cur_->stamp()) << ", mappoints "
             << frame_cur_->numFeatures() << ", interval " << dt;
    } else if (dt > TRACK_MAX_INTERVAl) {
        // 普通观测帧, 非关键帧
        keyframe_state = KEYFRAME_REMOVE_SECOND_NEW;
        LOGI << "Keyframe at " << Logging::doubleData(frame_cur_->stamp()) << " due to long interval";
    }

    // 切换上一关键帧, 用于时间间隔计算
    if (keyframe_state != KEYFRAME_NONE) {
        last_keyframe_ = frame_cur_;
    }

    if ((keyframe_state == KEYFRAME_REMOVE_OLDEST) || (keyframe_state == KEYFRAME_NORMAL)) {
        for (auto &mappoint : tracked_mappoint_) {
            // 更新路标点在观测帧中的使用次数
            mappoint->increaseUsedTimes();
        }
    }

    return keyframe_state;
}

bool Tracking::doResetTracking(bool is_reserved) {
    if (!frame_cur_->numFeatures()) {
        is_initializing_ = true;
        pts2d_new_.clear();
        pts2d_ref_.clear();
        pts2d_ref_frame_.clear();
        velocity_ref_.clear();

        if (is_reserved) {
            frame_ref_ = frame_cur_;

            // 当前帧为关键帧
            frame_cur_->setKeyFrame(KEYFRAME_NORMAL);
            is_new_keyframe_ = true;

            // 提取特征点
            featuresDetection(false);
        } else {
            frame_ref_ = nullptr;
        }

        return true;
    }

    return false;
}

double Tracking::relativeTranslation() {
    return (frame_cur_->pose().t - frame_ref_->pose().t).norm();
}

double Tracking::relativeRotation() {
    Eigen::AngleAxisd angleaxis(frame_cur_->pose().R.transpose() * frame_ref_->pose().R);

    return fabs(angleaxis.angle());
}

void Tracking::showTracking() {
    if (!is_use_visualization_) {
        return;
    }

    visual_drawer_->updateFrame(frame_cur_);
}

bool Tracking::depthAssociationProjection(PointCloud &pointcloud) {

    TimeCost timecost;

    // 点云数据投影降采样到一个距离图像, 仅保留最近距离的点, 剔除当前不可视的点, 并取前景点云

    size_t point_size = pointcloud.size();
    vector<float> range_image(ASSOCIATE_IMAGE_SIZE * ASSOCIATE_IMAGE_SIZE, FLT_MAX);
    vector<PointType> points_valid(ASSOCIATE_IMAGE_SIZE * ASSOCIATE_IMAGE_SIZE);
    for (size_t k = 0; k < point_size; k++) {
        const auto &point = pointcloud[k];

        // 不在图像中的点云
        if ((point.z < 0) || (fabs(point.x / point.z) > ASSOCIATE_POINT_TAN_VALUE) ||
            (fabs(point.y / point.z) > ASSOCIATE_POINT_TAN_VALUE)) {
            continue;
        }

        // top -> bottom
        double row_angle =
            atan2(point.y, sqrt(point.x * point.x + point.z * point.z)) + (M_PI * 0.5 - ASSOCIATE_IMAGE_FOV * 0.5);
        // right -> left
        double col_angle = atan2(point.z, point.x) - (M_PI * 0.5 - ASSOCIATE_IMAGE_FOV * 0.5);

        // 四舍五入
        int row = static_cast<int>(round(row_angle / ASSOCIATE_RANGE_IMAGE_ANGLE));
        int col = static_cast<int>(round(col_angle / ASSOCIATE_RANGE_IMAGE_ANGLE));

        if ((row < 0) || (col < 0) || (row >= ASSOCIATE_IMAGE_SIZE) || (col >= ASSOCIATE_IMAGE_SIZE)) {
            continue;
        }

        float distance = point.getVector3fMap().norm();
        int index      = col + row * ASSOCIATE_IMAGE_SIZE;
        if (distance < range_image[index]) {
            range_image[index]  = distance;
            points_valid[index] = pointcloud[k];
        }
    }

    // 抽取降采样的点云投影到单位球相机坐标系

    unit_sphere_pointcloud_ = PointCloudPtr(new PointCloud);
    for (int i = 0; i < ASSOCIATE_IMAGE_SIZE; i++) {
        for (int j = 0; j < ASSOCIATE_IMAGE_SIZE; j++) {
            int index = j + i * ASSOCIATE_IMAGE_SIZE;
            if (range_image[index] != FLT_MAX) {

                PointType p;
                // 强度数据保存距离信息
                p.intensity        = points_valid[index].getVector3fMap().norm();
                p.getVector3fMap() = points_valid[index].getVector3fMap().normalized();

                unit_sphere_pointcloud_->push_back(p);
            }
        }
    }

    if (unit_sphere_pointcloud_->empty()) {
        return false;
    }

    // 建立KD树
    kdtree_ = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    kdtree_->setInputCloud(unit_sphere_pointcloud_);

    LOGI << "Depth association projection " << unit_sphere_pointcloud_->size() << " points cost "
         << timecost.costInMillisecond() << " ms";

    return !unit_sphere_pointcloud_->empty();
}

int Tracking::depthAssociationReference() {
    // 无跟踪上的特征
    if (pts2d_cur_.empty()) {
        return 0;
    }

    TimeCost timecost;

    // 视觉特征投影到单位球相机坐标系

    // 参考帧的跟踪上的未三角化特征点
    auto pts2d_ref_undis = pts2d_ref_;
    auto pts2d_cur_undis = pts2d_cur_;

    camera_->undistortPoints(pts2d_ref_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    PointCloud unit_sphere_features;
    for (const auto &pts2d : pts2d_ref_undis) {
        PointType point;
        Vector3d puc           = camera_->pixel2unitcam(pts2d);
        point.getVector3fMap() = puc.cast<float>();
        point.intensity        = 0;

        unit_sphere_features.push_back(point);
    }

    int num_succeed = 0;
    vector<MapPointType> mappoint_types;
    for (size_t k = 0; k < unit_sphere_features.size(); k++) {
        // 非当前参考帧, 跳过
        if (pts2d_ref_frame_[k] != frame_ref_) {
            mappoint_types.push_back(MAPPOINT_NONE);
            continue;
        }
        auto &point = unit_sphere_features.points[k];

        auto type = depthAssociationPlaneFitting(point);

        mappoint_types.push_back(type);
        if (type != MAPPOINT_NONE) {
            num_succeed++;
        }
    }

    if (!num_succeed) {
        LOGW << "No associated depth and feature at " << Logging::doubleData(frame_ref_->stamp());
        return 0;
    }

    // 深度关联, 建立新的路标点
    int num_best       = 0;
    int num_associated = 0;
    vector<uint8_t> status;
    for (size_t k = 0; k < unit_sphere_features.size(); k++) {

        auto frame_ref = pts2d_ref_frame_[k];
        auto &point    = unit_sphere_features.points[k];
        auto depth     = point.intensity;

        // 未有效关联, 非当前参考帧, 深度无效
        if ((mappoint_types[k] == MAPPOINT_NONE) || (frame_ref != frame_ref_) ||
            !VisualMapPoint::isGoodDepth(depth, 1.0)) {
            status.push_back(1);
            continue;
        }

        Vector3d pc(point.x, point.y, point.z);
        // 创建新的路标点
        auto pw       = camera_->cam2world(pc, frame_ref_->pose());
        auto mappoint = VisualMapPoint::createMapPoint(frame_ref_, pw, pts2d_ref_undis[k], depth, mappoint_types[k]);

        auto feature = VisualFeature::createFeature(frame_cur_, velocity_cur_[k], pts2d_cur_undis[k], pts2d_cur_[k],
                                                    VisualFeature::FEATURE_DEPTH_ASSOCIATED);
        mappoint->addObservation(feature);
        feature->addMapPoint(mappoint);
        frame_cur_->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        feature = VisualFeature::createFeature(frame_ref_, velocity_ref_[k], pts2d_ref_undis[k], pts2d_ref_[k],
                                               VisualFeature::FEATURE_DEPTH_ASSOCIATED);
        mappoint->addObservation(feature);
        feature->addMapPoint(mappoint);
        frame_ref_->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        // 新路标点缓存到最新的关键帧, 不直接加入地图
        frame_cur_->addNewUnupdatedMappoint(mappoint);

        status.push_back(0);
        num_associated++;

        if (mappoint_types[k] == MAPPOINT_DEPTH_ASSOCIATED) {
            num_best++;
        }
    }

    // 未关联的特征
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(velocity_ref_, status);
    reduceVector(velocity_cur_, status);

    pts2d_new_ = pts2d_cur_;

    LOGI << "Associate reference features cost " << timecost.costInMillisecond() << " ms with " << num_succeed
         << " succeed " << num_associated << " associated with " << num_best << " best and left " << pts2d_cur_.size();

    return num_associated;
}

int Tracking::depthAssociationMapPoint() {
    if (mappoint_matched_.empty()) {
        return 0;
    }

    TimeCost timecost;

    vector<cv::Point2f> pts2d_map_undis;
    vector<MapPointType> mappoint_type;
    auto features = frame_ref_->features();
    for (const auto &mappoint : mappoint_matched_) {
        pts2d_map_undis.push_back(features[mappoint->id()]->keyPoint());
    }

    PointCloud unit_sphere_features;
    for (const auto &pts2d : pts2d_map_undis) {
        PointType point;
        Vector3d puc           = camera_->pixel2unitcam(pts2d);
        point.getVector3fMap() = puc.cast<float>();
        point.intensity        = 0;

        unit_sphere_features.push_back(point);
    }

    // 最近邻搜索并进行平面拟合计算深度
    vector<int> k_indices;
    vector<float> k_sqr_distances;

    int num_succeed = 0;
    vector<MapPointType> mappoint_types;
    for (size_t k = 0; k < unit_sphere_features.size(); k++) {
        if (mappoint_matched_[k]->mapPointType() == MAPPOINT_DEPTH_ASSOCIATED) {
            mappoint_types.push_back(MAPPOINT_NONE);
            continue;
        }

        auto &point = unit_sphere_features[k];

        auto type = depthAssociationPlaneFitting(point);
        mappoint_types.push_back(type);
        if (type != MAPPOINT_NONE) {
            num_succeed++;
        }
    }

    if (!num_succeed) {
        LOGW << "No associated depth and feature at " << Logging::doubleData(frame_ref_->stamp());
        return 0;
    }

    // 深度关联, 重设已有路标点
    int num_best       = 0;
    int num_associated = 0;
    for (size_t k = 0; k < unit_sphere_features.size(); k++) {
        const auto &point = unit_sphere_features.points[k];
        auto depth        = point.intensity;

        if ((mappoint_matched_[k]->mapPointType() == MAPPOINT_DEPTH_ASSOCIATED) ||
            (mappoint_types[k] == MAPPOINT_NONE) || !VisualMapPoint::isGoodDepth(depth, 1.0)) {
            continue;
        }

        Vector3d pc(point.x, point.y, point.z);
        auto pw       = camera_->cam2world(pc, frame_ref_->pose());
        auto mappoint = mappoint_matched_[k];

        // 重设路标点的参考帧
        mappoint->setReferenceFrame(frame_ref_, pw, pts2d_map_undis[k], depth, mappoint_types[k]);

        num_associated++;

        if (mappoint_types[k] == MAPPOINT_DEPTH_ASSOCIATED) {
            num_best++;
        }
    }

    LOGI << "Associate mappoint features cost " << timecost.costInMillisecond() << " ms with " << num_succeed
         << " succeed " << num_associated << " associated and " << num_best << " best";

    return num_associated;
}

MapPointType Tracking::depthAssociationPlaneFitting(PointType &point) {
    vector<int> k_indices;
    vector<float> k_sqr_distances;

    MapPointType type = MAPPOINT_NONE;

    // 最近邻搜索
    kdtree_->nearestKSearch(point, 5, k_indices, k_sqr_distances);

    if (k_sqr_distances[4] > ASSOCIATE_NEAREST_SQUARE_DISTANCE_THRESHOLD) {
        return type;
    }

    vector<Eigen::Vector3d> points;
    for (int k = 0; k < 5; k++) {
        const auto &pt = unit_sphere_pointcloud_->points[k_indices[k]];
        points.emplace_back(pt.getVector3fMap().cast<double>() * pt.intensity);
    }

    // 构建正定方程求解平面法向量
    Eigen::Matrix<double, 5, 3> ma;
    Eigen::Matrix<double, 5, 1> mb = -Eigen::Matrix<double, 5, 1>::Ones();
    for (int k = 0; k < 5; k++) {
        ma.row(k) = points[k];
    }
    Eigen::Vector3d mx = ma.colPivHouseholderQr().solve(mb);

    // 归一化处理
    double norm_inverse = 1.0 / mx.norm();
    mx.normalize();

    // 校验平面
    bool isvalid = true;
    vector<double> threshold;
    for (int k = 0; k < 5; k++) {
        double sm = fabs(mx.dot(points[k]) + norm_inverse);
        threshold.push_back(sm);
        if (sm > 0.1) {
            isvalid = false;
            break;
        }
    }

    double depth_diff = fabs(unit_sphere_pointcloud_->points[k_indices[0]].intensity -
                             unit_sphere_pointcloud_->points[k_indices[4]].intensity);
    if (depth_diff > 0.5) {
        isvalid = false;
    }
    if (isvalid) {
        // 选择距离最远的点构建方程求解深度
        Eigen::Vector3f vr = points[4].cast<float>();
        Eigen::Vector3f vi = point.getVector3fMap();
        Eigen::Vector3f vn = mx.cast<float>();

        float t = vn.dot(vr) / vn.dot(vi);

        // 有效特征深度, nan比较总是返回false
        if (std::isnan(t) || (t < VisualMapPoint::NEAREST_DEPTH) || (t > VisualMapPoint::FARTHEST_DEPTH)) {
            return type;
        }

        // 相机坐标系下的绝对坐标
        point.getVector3fMap() *= t;

        // 深度信息
        point.intensity = point.z;
        type            = MAPPOINT_DEPTH_ASSOCIATED;
    }

    return type;
}

bool Tracking::trackMappoint() {

    // 上一帧中的路标点
    mappoint_matched_.clear();
    vector<cv::Point2f> pts2d_map, pts2d_matched, pts2d_map_undis;
    vector<MapPointType> mappoint_type;
    auto features = frame_pre_->features();
    auto dt       = frame_cur_->stamp() - frame_pre_->stamp();
    for (auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (mappoint && !mappoint->isOutlier()) {
            mappoint_matched_.push_back(mappoint);
            pts2d_map_undis.push_back(feature.second->keyPoint());
            pts2d_map.push_back(feature.second->distortedKeyPoint());
            mappoint_type.push_back(mappoint->mapPointType());

            // INS预测的特征点
            auto pixel = camera_->world2pixel(mappoint->pos(), frame_cur_->pose());
            pts2d_matched.emplace_back(pixel);
        }
    }
    if (pts2d_matched.empty()) {
        LOGE << "No feature with mappoint in previous frame";
        return false;
    }

    // 预测的特征点像素坐标添加畸变, 用于跟踪
    camera_->distortPoints(pts2d_matched);

    vector<uint8_t> status, status_reverse;
    vector<float> error;
    vector<cv::Point2f> pts2d_reverse = pts2d_map;

    // 正向光流
    cv::calcOpticalFlowPyrLK(
        frame_pre_->image(), frame_cur_->image(), pts2d_map, pts2d_matched, status, error, cv::Size(21, 21), 1,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, TRACK_MAX_ITERATION, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // 反向光流
    cv::calcOpticalFlowPyrLK(
        frame_cur_->image(), frame_pre_->image(), pts2d_matched, pts2d_reverse, status_reverse, error, cv::Size(21, 21),
        1, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, TRACK_MAX_ITERATION, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // 跟踪失败的
    for (size_t k = 0; k < status.size(); k++) {
        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_matched[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_map[k]) < 0.5)) {
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    reduceVector(pts2d_map, status);
    reduceVector(pts2d_matched, status);
    reduceVector(mappoint_matched_, status);
    reduceVector(mappoint_type, status);
    reduceVector(pts2d_map_undis, status);

    if (pts2d_matched.empty()) {
        LOGE << "Track previous with mappoint failed at " << Logging::doubleData(frame_cur_->stamp());
        // 清除上一帧的跟踪
        if (is_use_visualization_) {
            visual_drawer_->updateTrackedMapPoints({}, {}, {});
        }
        return false;
    }

    // 匹配后的点, 需要重新矫正畸变
    auto pts2d_matched_undis = pts2d_matched;
    camera_->undistortPoints(pts2d_matched_undis);

    // 匹配的3D-2D
    frame_cur_->clearFeatures();
    tracked_mappoint_.clear();
    for (size_t k = 0; k < pts2d_matched_undis.size(); k++) {
        auto mappoint = mappoint_matched_[k];

        // 将3D-2D匹配到的landmarks指向到当前帧
        auto velocity = (camera_->pixel2cam(pts2d_matched_undis[k]) - camera_->pixel2cam(pts2d_map_undis[k])) / dt;
        auto feature  = VisualFeature::createFeature(frame_cur_, {velocity.x(), velocity.y()}, pts2d_matched_undis[k],
                                                     pts2d_matched[k], VisualFeature::FEATURE_MATCHED);
        mappoint->addObservation(feature);
        feature->addMapPoint(mappoint);
        frame_cur_->addFeature(mappoint->id(), feature);

        // 用于更新使用次数
        tracked_mappoint_.push_back(mappoint);
    }

    // 路标点跟踪情况
    if (is_use_visualization_) {
        visual_drawer_->updateTrackedMapPoints(pts2d_map, pts2d_matched, mappoint_type);
    }

    // LOGI << "Track " << tracked_mappoint_.size() << " map points";

    return true;
}

bool Tracking::trackReferenceFrame() {

    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame " << Logging::doubleData(frame_cur_->stamp());
        return false;
    }

    // 原始畸变补偿
    auto pts2d_new_undis = pts2d_new_;
    camera_->undistortPoints(pts2d_new_undis);

    // 补偿旋转预测
    Matrix3d r_cur_pre = frame_cur_->pose().R.transpose() * frame_pre_->pose().R;
    double dt          = frame_cur_->stamp() - frame_pre_->stamp();

    pts2d_cur_.clear();
    for (size_t k = 0; k < pts2d_new_undis.size(); k++) {
        auto pixel      = pts2d_new_undis[k];
        Vector3d pc_pre = camera_->pixel2cam(pixel);
        Vector3d pc_cur = r_cur_pre * pc_pre;

        pixel = camera_->cam2pixel(pc_cur);
        pts2d_cur_.emplace_back(pixel);
    }

    // 添加畸变
    camera_->distortPoints(pts2d_cur_);

    // 跟踪参考帧
    vector<uint8_t> status, status_reverse;
    vector<float> error;
    vector<cv::Point2f> pts2d_reverse = pts2d_new_;

    // 正向光流
    cv::calcOpticalFlowPyrLK(
        frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status, error, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, TRACK_MAX_ITERATION, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    // 反向光流
    cv::calcOpticalFlowPyrLK(
        frame_cur_->image(), frame_pre_->image(), pts2d_cur_, pts2d_reverse, status_reverse, error, cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, TRACK_MAX_ITERATION, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // 剔除跟踪失败的, 正向反向跟踪在0.5个像素以内
    for (size_t k = 0; k < status.size(); k++) {
        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_cur_[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_new_[k]) < 0.5)) {
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(pts2d_new_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(velocity_ref_, status);

    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame";
        visual_drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 原始带畸变的角点
    pts2d_new_undis      = pts2d_new_;
    auto pts2d_cur_undis = pts2d_cur_;
    camera_->undistortPoints(pts2d_new_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算像素速度
    velocity_cur_.clear();
    for (size_t k = 0; k < pts2d_cur_undis.size(); k++) {
        Vector3d vel      = (camera_->pixel2cam(pts2d_cur_undis[k]) - camera_->pixel2cam(pts2d_new_undis[k])) / dt;
        Vector2d velocity = {vel.x(), vel.y()};
        velocity_cur_.push_back(velocity);

        // 在关键帧后新增加的特征
        if (pts2d_ref_frame_[k]->id() > frame_ref_->id()) {
            velocity_ref_[k] = velocity;
        } else if (velocity_ref_[k].isZero()) {
            velocity_ref_[k] = velocity;
        }
    }

    // 计算视差
    auto pts2d_ref_undis = pts2d_ref_;
    camera_->undistortPoints(pts2d_ref_undis);

    // Fundamental粗差剔除
    if (pts2d_cur_.size() >= 15) {
        cv::findFundamentalMat(pts2d_new_undis, pts2d_cur_undis, cv::FM_RANSAC, reprojection_error_std_, 0.99, status);

        reduceVector(pts2d_ref_, status);
        reduceVector(pts2d_cur_, status);
        reduceVector(pts2d_ref_frame_, status);
        reduceVector(velocity_cur_, status);
        reduceVector(velocity_ref_, status);
    }

    if (pts2d_cur_.empty()) {
        LOGW << "No new feature in previous frame";
        if (is_use_visualization_) {
            visual_drawer_->updateTrackedRefPoints({}, {});
        }
        return false;
    }

    // 从参考帧跟踪过来的新特征点
    if (is_use_visualization_) {
        visual_drawer_->updateTrackedRefPoints(pts2d_ref_, pts2d_cur_);
    }

    // 用于下一帧的跟踪
    pts2d_new_ = pts2d_cur_;

    // LOGI << "Track " << pts2d_new_.size() << " reference points";

    return !pts2d_new_.empty();
}

void Tracking::featuresDetection(bool ismask) {
    // TimeCost timecost;

    // 特征点足够则无需提取
    int num_features = static_cast<int>(frame_cur_->features().size() + pts2d_ref_.size());
    if (num_features > (track_max_features_ - 5)) {
        return;
    }

    // 初始化分配内存
    int features_cnts[block_cnts_];
    vector<vector<cv::Point2f>> block_features(block_cnts_);
    // 必要的分配内存, 否则并行会造成数据结构错乱
    for (auto &block : block_features) {
        block.reserve(track_max_block_features_);
    }
    for (int k = 0; k < block_cnts_; k++) {
        features_cnts[k] = 0;
    }

    // 计算每个分块已有特征点数量
    int col, row;
    for (const auto &feature : frame_cur_->features()) {
        col = int(feature.second->keyPoint().x / (float) block_indexs_[0].first);
        row = int(feature.second->keyPoint().y / (float) block_indexs_[0].second);
        features_cnts[row * block_cols_ + col]++;
    }
    for (auto &pts2d : pts2d_new_) {
        col = int(pts2d.x / (float) block_indexs_[0].first);
        row = int(pts2d.y / (float) block_indexs_[0].second);
        features_cnts[row * block_cols_ + col]++;
    }

    // 设置感兴趣区域, 没有特征的区域
    Mat mask = Mat(camera_->size(), CV_8UC1, 255);
    if (ismask) {
        // 已经跟踪上的点
        for (auto &pt : frame_cur_->features()) {
            cv::circle(mask, pt.second->keyPoint(), track_min_pixel_distance_, 0, cv::FILLED);
        }

        // 还在跟踪的点
        for (const auto &pts2d : pts2d_new_) {
            cv::circle(mask, pts2d, track_min_pixel_distance_, 0, cv::FILLED);
        }
    }

    // 亚像素角点提取参数
    cv::Size win_size  = cv::Size(5, 5);
    cv::Size zero_zone = cv::Size(-1, -1);
    auto term_crit     = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01);

    tbb::parallel_for(int(0), block_cnts_, [&](const int &k) {
        int blocl_track_num = track_max_block_features_ - features_cnts[k];
        if (blocl_track_num > 0) {

            int cols = k % block_cols_;
            int rows = k / block_cols_;

            int col_sta = cols * block_indexs_[0].first;
            int col_end = col_sta + block_indexs_[0].first;
            int row_sta = rows * block_indexs_[0].second;
            int row_end = row_sta + block_indexs_[0].second;
            if (k != (block_cnts_ - 1)) {
                col_end -= 5;
                row_end -= 5;
            }

            auto block_image = frame_cur_->image().colRange(col_sta, col_end).rowRange(row_sta, row_end);
            auto block_mask  = mask.colRange(col_sta, col_end).rowRange(row_sta, row_end);

            cv::goodFeaturesToTrack(block_image, block_features[k], blocl_track_num, 0.01, track_min_pixel_distance_,
                                    block_mask);
            if (!block_features[k].empty()) {
                // 获取亚像素角点
                cv::cornerSubPix(block_image, block_features[k], win_size, zero_zone, term_crit);
            }
        }
    });

    // 调整角点的坐标
    int num_new_features = 0;
    for (int k = 0; k < block_cnts_; k++) {
        col = k % block_cols_;
        row = k / block_cols_;

        for (const auto &point : block_features[k]) {
            float x = static_cast<float>(col * block_indexs_[0].first) + point.x;
            float y = static_cast<float>(row * block_indexs_[0].second) + point.y;

            auto pts2d = cv::Point2f(x, y);
            pts2d_ref_.push_back(pts2d);
            pts2d_new_.push_back(pts2d);
            pts2d_ref_frame_.push_back(frame_cur_);
            velocity_ref_.emplace_back(0, 0);

            num_new_features++;
        }
    }

    // 提取新的特征
    // LOGI << "Add " << num_new_features << " new features to " << num_features;
}

bool Tracking::triangulation() {
    // 无跟踪上的特征
    if (pts2d_cur_.empty()) {
        return false;
    }

    Pose pose0;
    Pose pose1 = frame_cur_->pose();

    Eigen::Matrix<double, 3, 4> T_c_w_0, T_c_w_1;
    T_c_w_1 = pose2Tcw(pose1).topRows<3>();

    int num_succeeded = 0;
    int num_outlier   = 0;
    int num_reset     = 0;
    int num_outtime   = 0;

    // 原始带畸变的角点
    auto pts2d_ref_undis = pts2d_ref_;
    auto pts2d_cur_undis = pts2d_cur_;

    // 矫正畸变以进行三角化
    camera_->undistortPoints(pts2d_ref_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算使用齐次坐标, 相机坐标系
    vector<uint8_t> status;
    for (size_t k = 0; k < pts2d_cur_.size(); k++) {
        auto pp0 = pts2d_ref_undis[k];
        auto pp1 = pts2d_cur_undis[k];

        // 参考帧
        auto frame_ref = pts2d_ref_frame_[k];
        if (frame_ref->id() > frame_ref_->id()) {
            // 中途添加的特征, 修改参考帧
            pts2d_ref_frame_[k] = frame_cur_;
            pts2d_ref_[k]       = pts2d_cur_[k];
            status.push_back(1);
            num_reset++;
            continue;
        }

        // 移除长时间跟踪导致参考帧已经不在窗口内的观测, 避免移除当前参考帧
        if (visual_map_->isWindowNormal() && !visual_map_->isKeyFrameInMap(frame_ref)) {
            status.push_back(0);
            num_outtime++;
            continue;
        }

        // 进行必要的视差检查, 保证三角化有效
        pose0           = frame_ref->pose();
        double parallax = keyPointParallax(pts2d_ref_undis[k], pts2d_cur_undis[k], pose0, pose1);
        if (parallax < TRACK_MIN_PARALLAX) {
            status.push_back(1);
            continue;
        }
        T_c_w_0 = pose2Tcw(pose0).topRows<3>();

        // 三角化
        Vector3d pc0 = camera_->pixel2cam(pts2d_ref_undis[k]);
        Vector3d pc1 = camera_->pixel2cam(pts2d_cur_undis[k]);
        Vector3d pw;
        triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);

        // 三角化错误的点剔除
        if (!VisualMapPoint::isGoodToTrack(camera_, pp0, pose0, pw, reprojection_error_std_, 3.0) ||
            !VisualMapPoint::isGoodToTrack(camera_, pp1, pose1, pw, reprojection_error_std_, 3.0)) {
            status.push_back(0);
            num_outlier++;
            continue;
        }
        status.push_back(0);
        num_succeeded++;

        // 新的路标点, 加入新的观测, 路标点加入地图
        auto pc       = camera_->world2cam(pw, frame_ref->pose());
        double depth  = pc.z();
        auto mappoint = VisualMapPoint::createMapPoint(frame_ref, pw, pts2d_ref_undis[k], depth, MAPPOINT_TRIANGULATED);

        // 参考帧特征
        auto feature = VisualFeature::createFeature(frame_ref, velocity_ref_[k], pts2d_ref_undis[k], pts2d_ref_[k],
                                                    VisualFeature::FEATURE_TRIANGULATED);
        mappoint->addObservation(feature);
        feature->addMapPoint(mappoint);
        frame_ref->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        // 当前帧特征
        feature = VisualFeature::createFeature(frame_cur_, velocity_cur_[k], pts2d_cur_undis[k], pts2d_cur_[k],
                                               VisualFeature::FEATURE_TRIANGULATED);
        mappoint->addObservation(feature);
        feature->addMapPoint(mappoint);
        frame_cur_->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        // 新三角化的路标点缓存到最新的关键帧, 不直接加入地图
        frame_cur_->addNewUnupdatedMappoint(mappoint);
    }

    // 由于视差不够未及时三角化的角点
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(velocity_ref_, status);

    pts2d_new_ = pts2d_cur_;

    LOGI << "Triangulate " << num_succeeded << " 3D points with " << pts2d_cur_.size() << " left, " << num_reset
         << " reset, " << num_outtime << " outtime and " << num_outlier << " outliers";
    return true;
}

void Tracking::triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw) {
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();

    design_matrix.row(0) = pc0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = pc0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = pc1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = pc1[1] * pose1.row(2) - pose1.row(1);

    Eigen::Vector4d point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    pw                    = point.head<3>() / point(3);
}

template <typename T> void Tracking::reduceVector(T &vec, vector<uint8_t> status) {
    size_t index = 0;
    for (size_t k = 0; k < vec.size(); k++) {
        if (status[k]) {
            vec[index++] = vec[k];
        }
    }
    vec.resize(index);
}

double Tracking::ptsDistance(cv::Point2f &pt1, cv::Point2f &pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool Tracking::isOnBorder(const cv::Point2f &pts) {
    // 边界上的点
    return pts.x < 5.0 || pts.y < 5.0 || (pts.x > (camera_->width() - 5.0)) || (pts.y > (camera_->height() - 5.0));
}

Eigen::Matrix4d Tracking::pose2Tcw(const Pose &pose) {
    Eigen::Matrix4d Tcw;
    Tcw.setZero();
    Tcw(3, 3) = 1;

    Tcw.block<3, 3>(0, 0) = pose.R.transpose();
    Tcw.block<3, 1>(0, 3) = -pose.R.transpose() * pose.t;
    return Tcw;
}

double Tracking::keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0,
                                  const Pose &pose1) {
    Vector3d pc0 = camera_->pixel2cam(pp0);
    Vector3d pc1 = camera_->pixel2cam(pp1);

    // 补偿掉旋转
    Vector3d pc01 = pose1.R.transpose() * pose0.R * pc0;

    // 像素大小
    return (pc01.head<2>() - pc1.head<2>()).norm() * camera_->focalLength();
}

} // namespace visual

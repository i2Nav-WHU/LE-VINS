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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "visual/visual_map.h"

#include "factors/marginalization_info.h"
#include "preintegration/preintegration.h"

#include <ceres/ceres.h>

#include <deque>
#include <memory>

using std::deque;
using std::string;
using std::vector;

typedef IntegrationStateData StateData;
typedef std::shared_ptr<PreintegrationBase> PreBasePtr;
typedef ceres::ResidualBlockId ResidualId;

typedef vector<ResidualId> Residuals;
typedef vector<std::pair<ResidualId, bool>> ResidualOutliers;

class Optimizer {

public:
    typedef std::shared_ptr<Optimizer> Ptr;

    Optimizer() = delete;
    Optimizer(const string &configfile, std::shared_ptr<IntegrationParameters> parameters,
              PreintegrationOptions options, visual::VisualMap::Ptr visual_map, visual::Camera::Ptr camera);

    bool marginalization(deque<double> &timelist, deque<StateData> &statedatalist,
                         deque<PreBasePtr> &preintegrationlist);
    bool optimization(const deque<double> &timelist, deque<StateData> &statedatalist,
                      deque<PreBasePtr> &preintegrationlist);

    const vector<double> &reprojectionError();

    void constructPrior(const StateData &statedata, bool is_zero_velocity);

    static int getStateDataIndex(const deque<double> &timelist, double time);

private:
    void updateParametersFromOptimizer(const deque<double> &timelist, const deque<StateData> &statedatalist);
    bool visualLandmarkOutlierCulling();
    void addStateParameters(ceres::Problem &problem, deque<StateData> &statedatalist);
    void addVisualParameters(ceres::Problem &problem);
    void doReintegration(std::deque<PreBasePtr> &preintegrationlist, std::deque<StateData> &statedatalist);

    Residuals addReprojectionFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                     deque<StateData> &statedatalist);
    void addImuFactors(ceres::Problem &problem, deque<StateData> &statedatalist, deque<PreBasePtr> &preintegrationlist);
    static ResidualOutliers visualOutlierCullingByChi2(ceres::Problem &problem, Residuals &residual_ids);

public:
    void updateCameraExtrinsic(Pose &pose_b_c, double &td_b_c) const;

    const vector<Vector3d> &newFixedVisualMappoints() {
        return fixed_visual_mappoints_;
    }

    const vector<int> &iterations() {
        return iterations_;
    }

    const vector<double> &timecosts() {
        return timecosts_;
    }

    const vector<double> &outliers() {
        return outliers_;
    }

private:
    const double ZERO_VELOCITY_STD = 0.03; // 0.03 m/s
    double imudatarate_{200};

    std::shared_ptr<IntegrationParameters> integration_parameters_;
    PreintegrationOptions preintegration_options_;

    visual::VisualMap::Ptr visual_map_;
    visual::Camera::Ptr camera_;

    // 传感器使用
    bool is_use_lidar_depth_{false};

    // 边缘化
    std::shared_ptr<MarginalizationInfo> last_marginalization_info_{nullptr};
    vector<double *> last_marginalization_parameter_blocks_;
    vector<Vector3d> fixed_visual_mappoints_;

    // 估计参数
    std::unordered_map<ulong, double> invdepthlist_;
    double extrinsic_b_c_[8]{0};

    // 先验
    bool is_use_prior_{false};
    double mix_prior_[18];
    double mix_prior_std_[18];
    double pose_prior_[7];
    double pose_prior_std_[6];

    double extrinsic_cam_prior_[7];
    double extrinsic_cam_prior_std_[6];
    double gb_prior_std_;
    double ab_prior_std_;

    // 优化选项
    bool optimize_estimate_cam_extrinsic_;
    bool optimize_estimate_cam_td_;
    bool optimize_cam_extrinsic_accurate_;

    double optimize_reprojection_error_std_;
    size_t optimize_windows_size_;
    size_t optimize_half_windows_size_;

    double optimize_reprojection_std_;

    // 统计参数
    vector<int> iterations_;
    vector<double> timecosts_;
    vector<double> outliers_;
    vector<double> reprojection_;
};

#endif // OPTIMIZER_H

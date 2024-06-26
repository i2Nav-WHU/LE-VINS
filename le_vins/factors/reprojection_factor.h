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

#ifndef REPROJECTION_FACTOR_H
#define REPROJECTION_FACTOR_H

#include "common/rotation.h"

#include <Eigen/Geometry>
#include <ceres/ceres.h>

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

class ReprojectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1, 1> {

public:
    ReprojectionFactor() = delete;

    // 标准差为归一化相机下的重投影误差观测, pixel / f
    ReprojectionFactor(Vector3d pts0, Vector3d pts1, Vector3d vel0, Vector3d vel1, double td0, double td1, double std);

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;

private:
    // 归一化相机坐标系下的坐标
    Vector3d pts0_;
    Vector3d pts1_;

    Vector3d vel0_, vel1_;
    double td0_, td1_;

    Eigen::Matrix2d sqrt_info_;
};

#endif // REPROJECTION_FACTOR_H

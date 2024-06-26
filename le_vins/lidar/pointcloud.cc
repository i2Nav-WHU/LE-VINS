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

#include "lidar/pointcloud.h"

#include <Eigen/Eigenvalues>
#include <tbb/tbb.h>

namespace lidar {

PointType PointCloudCommon::relativeProjection(const Pose &pose0, const Pose &pose1, const PointType &point_l1) {
    Pose pose_l0_l1;
    pose_l0_l1.R = pose0.R.transpose() * pose1.R;
    pose_l0_l1.t = pose0.R.transpose() * (pose1.t - pose0.t);

    return pointProjection(pose_l0_l1, point_l1);
}

PointType PointCloudCommon::relativeProjectionToCamera(const Pose &pose_c_l, const Pose &pose_cam0,
                                                       const Pose &pose_cam1, const PointType &point_lidar) {
    PointType point_cam1 = pointProjection(pose_c_l, point_lidar);
    PointType point_cam0;

    Pose pose_cam0_cam1;
    pose_cam0_cam1.R = pose_cam0.R.transpose() * pose_cam1.R;
    pose_cam0_cam1.t = pose_cam0.R.transpose() * (pose_cam1.t - pose_cam0.t);

    point_cam0 = pointProjection(pose_cam0_cam1, point_cam1);

    return point_cam0;
}

void PointCloudCommon::pointCloudProjection(const Pose &pose, PointCloudPtr src, PointCloudPtr dst) {
    size_t points_size = src->size();
    dst->resize(points_size);

    auto projection_function = [&](const tbb::blocked_range<size_t> &range) {
        for (size_t k = range.begin(); k != range.end(); k++) {
            auto &point = src->points[k];

            dst->points[k] = pointProjection(pose, point);
        }
    };

    tbb::parallel_for(tbb::blocked_range<size_t>(0, points_size), projection_function);
}

PointType PointCloudCommon::pointProjection(const Pose &pose, const PointType &point) {
    PointType projected = point;

    projected.getVector3fMap() = (pose.R * point.getVector3fMap().cast<double>() + pose.t).cast<float>();

    return projected;
}

bool PointCloudCommon::planeEstimation(const PointVector &points, double threshold, Vector3d &unit_normal_vector,
                                       double &norm_inverse, std::vector<double> &distance) {
    if (points.size() < PLANE_ESTIMATION_POINT_SZIE) {
        return false;
    }

    // 构建超定方程求解法向量 A * n = b
    Eigen::Matrix<double, PLANE_ESTIMATION_POINT_SZIE, 3> A;
    Eigen::Matrix<double, PLANE_ESTIMATION_POINT_SZIE, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0;

    for (int k = 0; k < PLANE_ESTIMATION_POINT_SZIE; k++) {
        A.row(k) = points[k].getVector3fMap().cast<double>();
    }
    Vector3d normal_vector = A.colPivHouseholderQr().solve(b);

    // 法向量
    norm_inverse       = 1.0 / normal_vector.norm();
    unit_normal_vector = normal_vector * norm_inverse;

    // 校验点到平面距离
    distance.clear();
    for (int k = 0; k < PLANE_ESTIMATION_POINT_SZIE; k++) {
        auto dis = unit_normal_vector.dot(points[k].getVector3fMap().cast<double>()) + norm_inverse;
        distance.push_back(dis);
        if (fabs(dis) > threshold) {
            return false;
        }
    }
    return true;
}

double PointCloudCommon::pointSquareDistance(const PointType &point0, const PointType &point1) {
    return (point0.getVector3fMap() - point1.getVector3fMap()).squaredNorm();
}

double PointCloudCommon::pointToPlaneStd(const PointVector &nearest, const PointType &target, double point_std) {
    // 点的协方差固定
    Matrix3d point_var = Matrix3d::Identity();
    point_var.diagonal() *= point_std * point_std;

    size_t point_size         = nearest.size();
    double point_size_inverse = 1.0 / static_cast<double>(point_size);

    // 平面中心点和平面点协方差
    Matrix3d points_cov   = Matrix3d::Zero();
    Vector3d plane_center = Vector3d::Zero();
    for (const auto &point : nearest) {
        Vector3d pv = point.getVector3fMap().cast<double>();
        points_cov += pv * pv.transpose();
        plane_center += pv;
    }
    plane_center = plane_center * point_size_inverse;
    points_cov   = points_cov * point_size_inverse - plane_center * plane_center.transpose();

    // 平面点方差特征值分解
    Eigen::EigenSolver<Matrix3d> eigen_solver(points_cov);
    Eigen::Matrix3cd evecs = eigen_solver.eigenvectors();
    Eigen::Vector3cd evals = eigen_solver.eigenvalues();

    Vector3d evals_real = evals.real();
    Matrix3d evecs_real = evecs.real();

    Eigen::Matrix3f::Index evals_min, evals_max;
    evals_real.rowwise().sum().minCoeff(&evals_min);
    evals_real.rowwise().sum().maxCoeff(&evals_max);

    // int evals_mid = 3 - evals_min - evals_max;
    // Vector3d evec_min = evecs_real.col(evals_min);
    // Vector3d evec_mid = evecs_real.col(evals_mid);
    // Vector3d evec_max = evecs_real.col(evals_max);

    // 观测标准差
    double point_to_plane_std = 0;

    // 法向量
    Vector3d normal_vector = evecs_real.col(evals_min);

    Matrix3d jaco_center = Matrix3d::Identity();
    jaco_center.diagonal() *= point_size_inverse;

    // 法向量和中心点的协方差
    Eigen::Matrix<double, 6, 6> plane_cov;
    plane_cov.setZero();

    for (size_t i = 0; i < point_size; i++) {
        Vector3d pv = nearest[i].getVector3fMap().cast<double>();

        Eigen::Matrix<double, 6, 3> jaco;
        Matrix3d F = Matrix3d::Zero();
        for (int m = 0; m < 3; m++) {
            if (m != evals_min) {
                F.row(m) =
                    point_size_inverse * (pv - plane_center).transpose() / (evals_real[evals_min] - evals_real[m]) *
                    (evecs_real.col(m) * normal_vector.transpose() + normal_vector * evecs_real.col(m).transpose());
            }
        }

        jaco.block<3, 3>(0, 0) = evecs_real * F;
        jaco.block<3, 3>(3, 0) = jaco_center;
        plane_cov += jaco * point_var * jaco.transpose();
    }

    // 点到面方差
    Eigen::Matrix<double, 1, 6> jaco;
    jaco.block<1, 3>(0, 0) = target.getVector3fMap().cast<double>() - plane_center;
    jaco.block<1, 3>(0, 3) = -normal_vector.transpose();

    // 法向量和中心点
    double var = jaco * plane_cov * jaco.transpose();
    // 目标点协方差
    var += normal_vector.transpose() * point_var * normal_vector;
    point_to_plane_std = sqrt(var);

    return point_to_plane_std;
}

} // namespace lidar

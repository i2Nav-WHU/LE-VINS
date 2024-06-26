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

#include "preintegration/preintegration.h"

#include <ceres/ceres.h>

class ImuZeroVelocityFactor : public ceres::CostFunction {

public:
    ImuZeroVelocityFactor(PreintegrationOptions options, double vel_std)
        : options_(options)
        , vel_std_(vel_std) {
        *mutable_parameter_block_sizes() = vector<int>{Preintegration::numMixParameter(options_)};
        set_num_residuals(3);
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {

        // parameters: vel[3], bg[3], ba[3], sodo, abv

        // vel, bg, ba
        for (size_t k = 0; k < 3; k++) {
            residuals[k] = parameters[0][k] / vel_std_;
        }

        if (options_ == PREINTEGRATION_NORMAL) {
            if (jacobians && jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 9, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco.setZero();

                for (size_t k = 0; k < 3; k++) {
                    jaco(k, k) = 1.0 / vel_std_;
                }
            }
        } else if (options_ == PREINTEGRATION_ODO) {
            if (jacobians && jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 10, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco.setZero();

                for (size_t k = 0; k < 3; k++) {
                    jaco(k, k) = 1.0 / vel_std_;
                }
            }
        } 

        return true;
    }

private:
    PreintegrationOptions options_;

    double vel_std_;
};

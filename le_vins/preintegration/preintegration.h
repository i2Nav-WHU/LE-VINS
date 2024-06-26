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

#ifndef PREINTEGRATION_H
#define PREINTEGRATION_H

#include "preintegration/preintegration_base.h"
#include "preintegration/preintegration_normal.h"
#include "preintegration/preintegration_odo.h"

enum PreintegrationOptions {
    PREINTEGRATION_NORMAL = 0,
    PREINTEGRATION_ODO    = 1, // WITH_ODO
};

class Preintegration {

public:
    Preintegration() = default;

    static PreintegrationOptions getOptions(const IntegrationConfiguration &config)  {
        int options = 0;

        if (config.isuseodo) {
            options += WITH_ODO;
        }

        return static_cast<PreintegrationOptions>(options);
    }

    static std::shared_ptr<PreintegrationBase>
        createPreintegration(const std::shared_ptr<IntegrationParameters> &parameters, const IMU &imu0,
                             const IntegrationState &state, PreintegrationOptions options) {
        std::shared_ptr<PreintegrationBase> preintegration;

        if (options == PREINTEGRATION_NORMAL) {
            preintegration = std::make_shared<PreintegrationNormal>(parameters, imu0, state);
        } else if (options == PREINTEGRATION_ODO) {
            preintegration = std::make_shared<PreintegrationOdo>(parameters, imu0, state);
        }

        return preintegration;
    }

    static int numPoseParameter() {
        return PreintegrationBase::NUM_POSE;
    }

    static IntegrationStateData stateToData(const IntegrationState &state, PreintegrationOptions options) {
        if (options == PREINTEGRATION_NORMAL) {
            return PreintegrationNormal::stateToData(state);
        } else if (options == PREINTEGRATION_ODO) {
            return PreintegrationOdo::stateToData(state);
        }
        return {};
    }

    static IntegrationState stateFromData(const IntegrationStateData &data, PreintegrationOptions options) {
        if (options == PREINTEGRATION_NORMAL) {
            return PreintegrationNormal::stateFromData(data);
        } else if (options == PREINTEGRATION_ODO) {
            return PreintegrationOdo::stateFromData(data);
        }

        return {};
    }

    static int numMixParameter(PreintegrationOptions options) {
        int num = 0;
        if (options == PREINTEGRATION_NORMAL) {
            num = PreintegrationNormal::NUM_MIX;
        } else if (options == PREINTEGRATION_ODO) {
            num = PreintegrationOdo::NUM_MIX;
        }
        return num;
    }

private:
    enum Options {
        WITH_ODO = 1,
    };
};

#endif // PREINTEGRATION_H

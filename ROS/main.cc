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

#include "fusion.h"

#include "le_vins/common/logging.h"
#include "le_vins/common/timecost.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>

#include <absl/strings/numbers.h>

void sigintHandler(int sig);
void checkStateThread(std::shared_ptr<Fusion> fusion);

int main(int argc, char *argv[]) {
    // 配置文件路径
    string config_file(argv[1]);

    // 命令行参数控制日志输出到标准输出
    bool logtostderr = false;
    if (argc >= 3) {
        int res;
        if (absl::SimpleAtoi(argv[2], &res)) {
            logtostderr = (res == 1);
        }
    }

    // Glog初始化
    Logging::initialization(argv, logtostderr, true);

    // ROS节点初始化
    ros::init(argc, argv, "vins_node", ros::InitOption::NoSigintHandler | ros::InitOption::AnonymousName);

    // 注册信号处理函数
    std::signal(SIGINT, sigintHandler);

    auto fusion = std::make_shared<Fusion>();

    // 退出检测线程
    std::thread check_thread(checkStateThread, fusion);

    std::cout << "Fusion process is started..." << std::endl;

    // 进入消息循环
    TimeCost timecost;
    fusion->run(config_file);
    check_thread.join();
    LOGI << "VINS process costs " << timecost.costInSecond() << " seconds";

    // 关闭日志
    Logging::shutdownLogging();

    return 0;
}

void sigintHandler(int sig) {
    std::cout << "Terminate by Ctrl+C " << sig << std::endl;
    global_finished = true;
}

void checkStateThread(std::shared_ptr<Fusion> fusion) {
    std::cout << "Check thread is started..." << std::endl;

    auto fusion_ptr = std::move(fusion);
    while (!global_finished) {
        sleep(1);
    }

    // 退出VINS处理线程
    fusion_ptr->setFinished();
    std::cout << "VINS has been shutdown ..." << std::endl;

    // 关闭ROS
    ros::shutdown();
    std::cout << "ROS node has been shutdown ..." << std::endl;
}

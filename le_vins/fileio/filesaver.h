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

#ifndef FILESAVER_H
#define FILESAVER_H

#include <memory>

#include "fileio/filebase.h"

class FileSaver : public FileBase {

public:
    typedef std::shared_ptr<FileSaver> Ptr;

    FileSaver() = default;
    FileSaver(const string &filename, int columns, int filetype = TEXT);

    static FileSaver::Ptr create(const string &filename, int columns, int filetype = TEXT) {
        return std::make_shared<FileSaver>(filename, columns, filetype);
    }

    ~FileSaver();

    bool open(const string &filename, int columns, int filetype = TEXT);

    void dump(const vector<double> &data);
    void dumpn(const vector<vector<double>> &data);

private:
    void dump_(const vector<double> &data);
};

#endif // FILESAVER_H

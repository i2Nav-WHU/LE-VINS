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

#ifndef FILELOADER_H
#define FILELOADER_H

#include <memory>

#include "fileio/filebase.h"

class FileLoader : public FileBase {

public:
    typedef std::shared_ptr<FileLoader> Ptr;

    FileLoader() = default;
    FileLoader(const string &filename, int columns, int filetype = TEXT);

    ~FileLoader();

    bool open(const string &filename, int columns, int filetype = TEXT);

    vector<double> load();
    vector<vector<double>> loadn(int epochs);

    bool load(vector<double> &data);
    bool loadn(vector<vector<double>> &data, int epochs);

private:
    vector<double> data_;

    bool load_();
};

#endif // FILELOADER_H

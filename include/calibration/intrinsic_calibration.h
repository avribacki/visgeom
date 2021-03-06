/*
This file is part of visgeom.

visgeom is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

visgeom is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with visgeom.  If not, see <http://www.gnu.org/licenses/>.
*/ 

#pragma once

#include "generic_calibration.h"

template<template<typename> class Camera>
class IntrinsicCameraCalibration : GenericCameraCalibration<Camera>
{
private:
    vector<CalibrationData> monoCalibDataVec;
    
    using GenericCameraCalibration<Camera>::initializeIntrinsic;
    using GenericCameraCalibration<Camera>::extractGridProjection;
    using GenericCameraCalibration<Camera>::constructGrid;
    using GenericCameraCalibration<Camera>::estimateInitialGrid;
    using GenericCameraCalibration<Camera>::addIntrinsicResidual;
    using GenericCameraCalibration<Camera>::residualAnalysis;
        
public:


    //TODO chanche the file formatting
    bool initialize(const string & infoFileName)
    {
        cout << "### Initialize calibration data ###" << endl;
        if (not initializeIntrinsic(infoFileName, monoCalibDataVec)) return false;
        constructGrid();
        return true;
    }
    
    bool compute(vector<double> & intrinsic)
    {
        cout << "### Initial extrinsic estimation ###" << endl;

        if (monoCalibDataVec.size() == 0)
        {
            cout << "ERROR : none of images were accepted" << endl;
            return false;
        } 
        
        //initial board positions estimation
        for (auto & item : monoCalibDataVec)
        {
            estimateInitialGrid(intrinsic, item.projection, item.extrinsic); 
        }
        
        cout << "### Intrinsic parameters calibration ###" << endl;
        // Problem initialization
        Problem problem;
        for (auto & item : monoCalibDataVec)
        {
            addIntrinsicResidual(problem, intrinsic, item.projection, item.extrinsic);
        }
               
        //run the solver
        Solver::Options options;
        options.max_num_iterations = 500;
        options.function_tolerance = 1e-10;
        options.gradient_tolerance = 1e-10;
        options.parameter_tolerance = 1e-10;
//        options.minimizer_progress_to_stdout = true;
        Solver::Summary summary;
        Solve(options, &problem, &summary);
        cout << summary.FullReport() << endl;
        
    } 
    
    void residualAnalysis(vector<double> & intrinsic)
    {
        cout << "### Intrinsic caibration - residual analysis ###" << endl;
        residualAnalysis(intrinsic, monoCalibDataVec);
    }
};


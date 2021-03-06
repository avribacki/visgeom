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

/*
Depth-from-motion class for semidense depth estimation
*/

#pragma once


#include "std.h"
#include "ocv.h"
#include "eigen.h"
#include "utils/filter.h"
#include "geometry/geometry.h"
#include "camera/eucm.h"

#include "eucm_epipolar.h"
#include "curve_rasterizer.h"
#include "depth_map.h"
#include "epipolar_descriptor.h"
#include "epipoles.h"

//TODO add errorMax threshold
struct MotionStereoParameters
{
    int scale = 3;
    int descLength = 5;
    int gradientThresh = 10;
    int verbosity = 0;
    int uMargin = 25, vMargin = 25;  // RoI left upper corner
    int biasMax = 10;
    int dispMax = 48;
};

class MotionStereo
{
public:
    MotionStereo(const EnhancedCamera * cam1, 
        const EnhancedCamera * cam2, MotionStereoParameters params) :
        // initialize the members
        camera1(cam1->clone()),
        camera2(cam2->clone()),
        params(params)
    {
    }

    ~MotionStereo()
    {
        delete camera1;
        camera1 = NULL;
        delete camera2;
        camera2 = NULL;
    }    
    
    void setBaseImage(const Mat8u & image)
    {
        image.copyTo(img1);
        computeMask();
    }
    
    //TODO split into functions
    void computeDepth(Transformation<double> T12,
            Mat8u img2, DepthMap & depth)
    {
        if (params.verbosity > 0) cout << "MotionStereo::computeDepth" << endl;
        epipolarPtr = new EnhancedEpipolar(T12, camera1, camera2, 2000, params.verbosity);
        StereoEpipoles epipoles(camera1, camera2, T12);
        Transform12 = T12;
        // init the output mat
        //TODO think about overhead
        if (params.verbosity > 1) cout << "    initializing the depth map" << endl;
        
        if (params.verbosity > 1) cout << "    descriptor kernel selection" << endl;
        int LENGTH = params.descLength;
        int HALF_LENGTH = LENGTH / 2;
        
        // compute the weights for matching cost
        // TODO make a separate header and a cpp file miscellaneous
        vector<int> kernelVec, waveVec;
        const int NORMALIZER = initKernel(kernelVec, LENGTH);
        const int WAVE_NORM = initWave(waveVec, LENGTH);
        EpipolarDescriptor epipolarDescriptor(LENGTH, WAVE_NORM, waveVec.data(), {1, 2, 3});
        
        if (params.verbosity > 1) cout << "    computing the scan limits" << endl;
        // get uncertainty range reconstruction in the first frame
        
        // discard non-salient points
        depth.applyMask(maskMat);
        
        vector<int> idxVec;
        Vector3dVec minDistVec, maxDistVec;
        depth.reconstructUncertainty(idxVec, minDistVec, maxDistVec);
        Vector2dVec pointVec = depth.getPointVec(idxVec);
        // reproject them onto the second image
        Vector3dVec minDist2Vec, maxDist2Vec;
        T12.inverseTransform(minDistVec, minDist2Vec);
        T12.inverseTransform(maxDistVec, maxDist2Vec);
        cout << 111 << endl;
        Vector2dVec pointMinVec;
        Vector2dVec pointMaxVec;
        vector<bool> maskMinVec, maskMaxVec;
        camera2->projectPointCloud(minDist2Vec, pointMinVec, maskMinVec);
        camera2->projectPointCloud(maxDist2Vec, pointMaxVec, maskMaxVec);
        
        vector<bool> maskVec;
        for (int i = 0; i < maskMinVec.size(); i++)
        {
            maskVec.push_back(maskMinVec[i] and maskMaxVec[i]);
        }
        
        
        if (params.verbosity > 1) cout << "    core loop" << endl;
        for (int ptIdx = 0; ptIdx < minDistVec.size(); ptIdx++)
        {
            if (not maskVec[ptIdx])
            {
                depth.at(idxVec[ptIdx]) = 0;
                continue;
            }   
            
            // ### compute descriptor ###
            if (params.verbosity > 2) cout << "        compute descriptor" << endl;
            // get the corresponding rasterizer
            CurveRasterizer<int, Polynomial2> descRaster(round(pointVec[ptIdx]), epipoles.getFirstPx(),
                                                epipolarPtr->getFirst(minDistVec[ptIdx]));
            if (epipoles.firstIsInverted()) descRaster.setStep(-1);
            
            // compute the actual descriptor
            vector<uint8_t> descriptor;
            const int step = epipolarDescriptor.compute(img1, descRaster, descriptor);
            if (not epipolarDescriptor.goodResp() or step < 1)
            {
                depth.at(idxVec[ptIdx]) = 0;
                continue;
            }
            // ### find the best correspondence on img2 ###
            if (params.verbosity > 2) cout << "        sampling the second image" << endl;
            //sample the second image
            //TODO traverse the epipolar line in the opposit direction and respect the disparity limit
            Vector2i goal = round(pointMinVec[ptIdx]);
            CurveRasterizer<int, Polynomial2> raster(round(pointMaxVec[ptIdx]), goal,
                                                epipolarPtr->getSecond(minDistVec[ptIdx]));
            Vector2i diff = round(pointMaxVec[ptIdx]) - goal;
            const int distance = min(int(diff.norm()), params.dispMax);
            raster.steps(-HALF_LENGTH*step);
            vector<uint8_t> sampleVec;
            vector<int> uVec, vVec;
            const int margin = LENGTH*step - 1;
            uVec.reserve(distance + margin);
            vVec.reserve(distance + margin);
            sampleVec.reserve(distance + margin);
            for (int d = 0; d < distance + margin; d++, raster.step())
            {
                if (raster.v < 0 or raster.v >= img2.rows 
                    or raster.u < 0 or raster.u >= img2.cols) sampleVec.push_back(0);
                else sampleVec.push_back(img2(raster.v, raster.u));
                uVec.push_back(raster.u);
                vVec.push_back(raster.v);
            }
            
            if (params.verbosity > 2) cout << "        find the best candidate" << endl;
            //compute the error and find the best candidate
            int dBest = 0;
            int eBest = LENGTH*255;
            int sum1 = filter(kernelVec.begin(), kernelVec.end(), descriptor.begin(), 0);
//            cout << "ERROR CURVE " << step << endl;
            for (int d = 0; d < distance; d++)
            {
                int sum2 = filter(kernelVec.begin(), kernelVec.end(), sampleVec.begin() + d, 0);
                int bias = 0; // min(params.biasMax, max(-params.biasMax, (sum2 - sum1) / NORMALIZER));
                int acc =  biasedAbsDiff(kernelVec.begin(), kernelVec.end(),
                                    descriptor.begin(), sampleVec.begin() + d, bias, step);
//                cout << acc << endl;
                if (eBest > acc)
                {
                    eBest = acc;
                    dBest = d;
                }
            }
            
            //TODO make triangulation checks and 
            Vector3d X1, X2;
            triangulate(pointVec[ptIdx][0], pointVec[ptIdx][1], 
                    uVec[dBest + HALF_LENGTH], vVec[dBest + HALF_LENGTH], X1);
            triangulate(pointVec[ptIdx][0], pointVec[ptIdx][1], 
                    uVec[dBest + HALF_LENGTH + 1], vVec[dBest + HALF_LENGTH + 1], X2);
            depth.at(idxVec[ptIdx]) = X1.norm();
            depth.sigma(idxVec[ptIdx]) = (X2 - X1).norm() / 2;
        }
        
        delete epipolarPtr;
        epipolarPtr = NULL;
    }
    
   
    // TODO a lot of overlap with EnhancedStereo, think about merging them of deriving them
    bool triangulate(double x1, double y1, double x2, double y2, Vector3d & X)
    {
        if (params.verbosity > 3) cout << "EnhancedStereo::triangulate" << endl;
        //Vector3d v1n = v1 / v1.norm(), v2n = v2 / v2.norm();
        Vector3d v1, v2;
        if (not camera1->reconstructPoint(Vector2d(x1, y1), v1) or 
            not camera2->reconstructPoint(Vector2d(x2, y2), v2) )
        {
            if (params.verbosity > 2) 
            {
                cout << "    not reconstructed " << Vector2d(x1, y1).transpose(); 
                cout << " # " << Vector2d(x2, y2).transpose() << endl;
            }
            X = Vector3d(0, 0, 0);
            return false;
        }
        Vector3d t = Transform12.trans();
        v2 = Transform12.rotMat() * v2;
        if (params.verbosity > 3) 
        {
            cout << "    pt1: " << x1 << " " << y1 << endl;
            cout << "    x1: " << v1.transpose() << endl;
            cout << "    pt2: " << x2 << " " << y2 << endl;
            cout << "    x2: " << v2.transpose() << endl;
        }
        double v1v2 = v1.dot(v2);
        double v1v1 = v1.dot(v1);
        double v2v2 = v2.dot(v2);
        double tv1 = t.dot(v1);
        double tv2 = t.dot(v2);
        double delta = -v1v1 * v2v2 + v1v2 * v1v2;
        if (abs(delta) < 1e-10) // TODO the constant to be revised
        {
            if (params.verbosity > 2) 
            {
                cout << "    not triangulated " << abs(delta) << " " << (abs(delta) < 1e-10) << endl;
            }
            X = Vector3d(0, 0, 0);
            return false;
        }
        double l1 = (-tv1 * v2v2 + tv2 * v1v2)/delta;
        double l2 = (tv2 * v1v1 - tv1 * v1v2)/delta;
        X = (v1*l1 + t + v2*l2)*0.5;
        return true;
    }
 
    
    
private:
    EnhancedEpipolar * epipolarPtr;
    // based on the image gradient
    void computeMask()
    {
        Mat16s gradx, grady;
        Sobel(img1, gradx, CV_16S, 1, 0, 1);
        Sobel(img1, grady, CV_16S, 0, 1, 1);
        Mat16s gradAbs = abs(gradx) + abs(grady);
        GaussianBlur(gradAbs, gradAbs, Size(5, 5), 0, 0);
        Mat8u gradAbs8u;
        gradAbs.convertTo(gradAbs8u, CV_8U);
        threshold(gradAbs8u, maskMat, params.gradientThresh, 128, CV_THRESH_BINARY);
        
    }
    
    // pose of the first to the second camera
    Transformation<double> Transform12;
    EnhancedCamera *camera1, *camera2;
    
    Mat8u img1;    
    Mat8u maskMat; //TODO compute mask
    MotionStereoParameters params;
};


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
Semi-global block matching algorithm for non-rectified images
NOTE:
(u, v) is an image point 
(x, y) is a depth map point 
*/

#pragma once

#include "std.h"
#include "ocv.h"
#include "eigen.h"

#include "geometry/geometry.h"
#include "camera/eucm.h"

#include "curve_rasterizer.h"
#include "depth_map.h"
#include "eucm_epipolar.h"
#include "epipolar_descriptor.h"
#include "utils/scale_parameters.h"
#include "epipoles.h"

struct StereoParameters : public ScaleParameters
{
    // basic parameters
    int dispMax = 48; // maximum disparity
    int uMargin = 0, vMargin = 0;  // RoI left upper corner
    int width = -1, height = -1;  // RoI size
    int lambdaStep = 5;
    int lambdaJump = 32;
    int maxBias = 10;
    
    int verbosity = 0;
    int maxDepth = 100;
    
    bool useUVCache = true;
    
    // to be called before using
    void init()
    {
        u0 = uMargin + scale; 
        v0 = vMargin + scale;
        
        int uBR, vBR; //bottom right
            
        if (width > 0) uMax = u0 + width;
        else uBR = uMax - uMargin - scale;
        
        if (height > 0) vMax = v0 + height;
        else vBR = vMax - vMargin - scale;
        
        xMax = x(uBR) + 1;
        yMax = y(vBR) + 1;
    }
};

class EnhancedStereo
{
public:
    enum CameraIdx {CAMERA_1, CAMERA_2};
    
    EnhancedStereo(Transformation<double> T12, const EnhancedCamera * cam1,
            const EnhancedCamera * cam2, const StereoParameters & stereoParams) :
            // initialize members
            Transform12(T12), 
            camera1(cam1->clone()),
            camera2(cam2->clone()),
            params(stereoParams),
            epipolar(T12, cam1, cam2, 2500),
            epipoles(cam1, cam2, T12)
    { 
        assert(params.dispMax % 2 == 0);
        params.init();
        createBuffer();
        computeReconstructed();
        computeRotated();
        computePinf();
        if (params.useUVCache) computeUVCache();
    }
    
    ~EnhancedStereo()
    {
        delete camera1;
        camera1 = NULL;
        delete camera2;
        camera2 = NULL;
    }
    
    // precompute coordinates for different disparities to speedup the computation
    void computeUVCache();
    
    // An interface function
    void comuteStereo(const Mat8u & img1, const Mat8u & img2, DepthMap & depthMap);
    
    // An interface function
    void comuteStereo(const Mat8u & img1, const Mat8u & img2, Mat32f & depthMat);
    
    //// EPIPOLAR GEOMETRY
    
    // computes reconstVec -- reconstruction of every pixel of the first image
    void computeReconstructed();
    
    // computes reconstRotVec -- reconstVec rotated into the second frame
    void computeRotated();
       
    // computes pinfVec -- projections of all the reconstructed points from the first image
    // onto the second image as if they were at infinity
    void computePinf();
    
    // calculate the coefficients of the polynomials for all the 
    void computeEpipolarIndices();
    
    // TODO remove from this class
    // draws an epipolar line  on the right image that corresponds to (x, y) on the left image
    void traceEpipolarLine(int u, int v, Mat8u & out, CameraIdx camIdx);
    
    //// DYNAMIC PROGRAMMING
    void createBuffer();
       
    // fill up the error buffer using 2*S-1 pixs along epipolar lines as local desctiprtors
    void computeCurveCost(const Mat8u & img1, const Mat8u & img2);
    
    void computeDynamicProgramming();
    
    void computeDynamicStep(const int* inCost, const uint8_t * error, int * outCost);
    
    void reconstructDisparity();  // using the result of the dynamic programming
    
    // TODO implement
    void upsampleDisparity(const Mat8u & img1, Mat8u & disparityMat);
    
    //// MISCELLANEOUS
    
    // index of an object in a linear array corresponding to pixel [row, col] 
    int getLinearIndex(int x, int y) { return params.xMax*y + x; }
      
    CurveRasterizer<int, Polynomial2> getCurveRasteriser1(int idx);
    CurveRasterizer<int, Polynomial2> getCurveRasteriser2(int idx);
    
    // reconstruction
    bool triangulate(double u1, double v1, double u2, double v2, Vector3d & X);
    void computeDepth(Mat32f & distanceMat);
    
    //TODO put generatePlane elsewhere
    void generatePlane(Transformation<double> TcameraPlane, 
            Mat32f & distance, const Vector3dVec & polygonVec);
            
    //TODO put generatePlane elsewhere        
    void generatePlane(Transformation<double> TcameraPlane, 
            DepthMap & distance, const Vector3dVec & polygonVec);
            
    double computeDepth(int x, int y);
    bool computeDepth(int x, int y, double & dist, double & sigma);
    
    void fillGaps(uint8_t * const data, const int step);
    
    int getHalfLength() { return min(4, max(params.scale - 1, 1)); }
    
private:
    EnhancedEpipolar epipolar;
    StereoEpipoles epipoles;
    
    Transformation<double> Transform12;  // pose of camera 2 wrt camera 1
    EnhancedCamera *camera1, *camera2;
   
    std::vector<bool> maskVec;
    
    Vector2dVec pointVec1;  // the depth points on the image 1
    Vector3dVec reconstVec;  // reconstruction of every pixel by cam1
    Vector3dVec reconstRotVec;  // reconstVec rotated into the second frame
    Vector2dVec pinfVec;  // projection of reconstRotVec by cam2
    
    // discretized version
    Vector2iVec pointPxVec1;
    Vector2iVec pinfPxVec;
    
    
    const int DISPARITY_MARGIN = 20;
    Mat32s uCache, vCache;
    Mat8u errorBuffer;
    Mat32s tableauLeft, tableauRight; //FIXME check the type through the code
    Mat32s tableauTop, tableauBottom;
    Mat8u smallDisparity;
    
    StereoParameters params;
};


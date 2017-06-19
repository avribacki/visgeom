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

#include "io.h"
#include "ocv.h"
#include "eigen.h"

#include "geometry/geometry.h"
#include "camera/pinhole.h"
#include "camera/eucm.h"

#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ximgproc.hpp"

void initRemap(const array<double, 6> & params1, const array<double, 3> & params2,
    Mat32f & mapX, Mat32f & mapY, const array<double, 3> & rot)
{
    EnhancedCamera cam1(params1.data());
    Pinhole cam2(params2[0], params2[1], params2[2]);
    Vector2dVec imagePoints;
    mapX.create(params2[1]*2, params2[0]*2);
    mapY.create(params2[1]*2, params2[0]*2);
    for (unsigned int i = 0; i < mapX.rows; i++)
    {
        for (unsigned int j = 0; j < mapX.cols; j++)
        {
            imagePoints.push_back(Vector2d(j, i));
        }
    }
    Vector3dVec pointCloud;
    cam2.reconstructPointCloud(imagePoints, pointCloud);
    Transformation<double> T(0, 0, 0, rot[0], rot[1], rot[2]);
    T.transform(pointCloud, pointCloud);
    cam1.projectPointCloud(pointCloud, imagePoints);
    
    auto pointIter = imagePoints.begin();
    for (unsigned int i = 0; i < mapX.rows; i++)
    {
        for (unsigned int j = 0; j < mapX.cols; j++)
        {
            mapX(i, j) = (*pointIter)[0];
            mapY(i, j) = (*pointIter)[1];
            ++pointIter;
        }
    }
}

void initRemap2(const std::array<double, 6>& enhanced_params,
                const std::array<double, 3> & pinhole_params,
                cv::Mat1f& map_x, cv::Mat1f& map_y,
                const std::vector<double>& rotation_angles)
{
    double pinhole_u0 = pinhole_params[0];
    double pinhole_v0 = pinhole_params[1];
    double pinhole_f = pinhole_params[2];

    double alpha = enhanced_params[0];
    double beta = enhanced_params[1];
    double fu = enhanced_params[2];
    double fv = enhanced_params[3];
    double u0 = enhanced_params[4];
    double v0 = enhanced_params[5];

    map_x.create(pinhole_v0 * 2, pinhole_u0 * 2);
    map_y.create(pinhole_v0 * 2, pinhole_u0 * 2);

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_angles, rotation_matrix);

    for (unsigned int i = 0; i < map_x.rows; i++) {
        for (unsigned int j = 0; j < map_y.cols; j++) {
            // 1. Project pinhole image to 3D point.
            cv::Mat1d point(3, 1);

            point(0, 0) = (j - pinhole_u0) / pinhole_f;
            point(1, 0) = (i - pinhole_v0) / pinhole_f;
            point(2, 0) = 1.0;

            // 2. Rotate this point
            point = rotation_matrix * point;

            double x = point(0, 0);
            double y = point(1, 0);
            double z = point(2, 0);

            // 3. Project to conic surface
            double denom = alpha * std::sqrt(z*z + beta * (x*x + y*y)) + (1.0 - alpha) * z;

            // 4. Project the point to the mu plane
            double xn = x / denom;
            double yn = y / denom;

            // 5. Compute remap parameters
            map_x(i, j) = fu * xn + u0;
            map_y(i, j) = fv * yn + v0;
        }
    }
}

void fakeMovement(const std::array<double, 6>& enhanced_params,
                const std::array<double, 3> & pinhole_params,
                cv::InputArray input, cv::OutputArray output,
                const std::vector<double>& rotation_angles,
                const std::vector<double>& translation)
{
    cv::Mat1f map_x, map_y;

    double pinhole_u0 = pinhole_params[0];
    double pinhole_v0 = pinhole_params[1];
    double pinhole_f = pinhole_params[2];

    double alpha = enhanced_params[0];
    double beta = enhanced_params[1];
    double fu = enhanced_params[2];
    double fv = enhanced_params[3];
    double u0 = enhanced_params[4];
    double v0 = enhanced_params[5];

    map_x.create(pinhole_v0 * 2, pinhole_u0 * 2);
    map_y.create(pinhole_v0 * 2, pinhole_u0 * 2);

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_angles, rotation_matrix);

    for (unsigned int i = 0; i < map_x.rows; i++) {
        for (unsigned int j = 0; j < map_y.cols; j++) {
            // 1. Project pinhole image to 3D point.
            cv::Mat1d point(3, 1);

            point(0, 0) = (j - pinhole_u0) / pinhole_f;
            point(1, 0) = (i - pinhole_v0) / pinhole_f;
            point(2, 0) = 1.0;

            // 2. Rotate this point
            point = rotation_matrix * point;

            // 3. Translate this point
            double x = point(0, 0) + translation[0];
            double y = point(1, 0) + translation[1];
            double z = point(2, 0) + translation[2];

            // 3. Project to conic surface
            double denom = alpha * std::sqrt(z*z + beta * (x*x + y*y)) + (1.0 - alpha) * z;

            // 4. Project the point to the mu plane
            double xn = x / denom;
            double yn = y / denom;

            // 5. Compute remap parameters
            map_x(i, j) = fu * xn + u0;
            map_y(i, j) = fv * yn + v0;
        }
    }

    cv::remap(input, output, map_x, map_y, cv::INTER_LINEAR);
}


cv::Mat findCeiling(cv::Mat source)
{
    cv::Mat result, with_color;

//    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), Point(3, 3));
//    cv::morphologyEx(source, result, cv::MORPH_OPEN, element);
    cv::blur(source, result, cv::Size(3,3));
    cv::Canny(result, result, 20, 100, 3);
    cv::cvtColor(result, with_color, cv::COLOR_GRAY2BGR);

#if 0
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(result, lines, 1, CV_PI/180, 100);

    for (const cv::Vec2f l : lines) {
        float rho = l[0];
        float theta = l[1];

        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a*rho, y0 = b*rho;

        cv::Point pt1(cvRound(x0 + 1000 * (-b)),
                      cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)),
                      cvRound(y0 - 1000 * (a)));
        cv::line(with_color, pt1, pt2, Scalar(0,0,255), 3, 8);
    }
#else
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(result, lines, 1, CV_PI/180, 80, 30, 10);
    for (const cv::Vec4i l : lines) {
        cv::line(with_color, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, 8);
    }
#endif

    cv::imshow("open", with_color);
    return result;
}

int main(int argc, char** argv) {

    // read parameters and image names
    array<double, 6> params;
    ifstream paramFile(argv[1]);
    cout << argv[1] << endl;
    cout << "EU Camera model parameters :" << endl;
    for (auto & p: params) 
    {
        paramFile >> p;
        cout << setw(10) << p;
    }
    cout << endl;
    paramFile.ignore();
    
    array<double, 3> params2;
    cout << "Pinhole rectified camera parameters :" << endl;
    for (auto & p: params2) 
    {
        paramFile >> p;
        cout << setw(10) << p;
    }
    cout << endl;
    paramFile.ignore();
    
    std::vector<double> rotation(3);
    std::cout << "Rotation of the pinhole camera :" << std::endl;
    for (auto & p: rotation) 
    {
        paramFile >> p;
        cout << setw(10) << p;
    }
    cout << endl;
    paramFile.ignore();
    
    Mat32f  mapX, mapY;
    initRemap2(params, params2, mapX, mapY, rotation);
    
    string dirName, fileName;
    getline(paramFile, dirName);

    std::vector<std::string> file_list;

    while (getline(paramFile, fileName)) {
        file_list.emplace_back(fileName);
    }

//    for (int i = 10; i < file_list.size(); ++i)
//    {

        cv::Mat img1, original1 = cv::imread(dirName + file_list[0], 0);
        cv::Mat img2, original2 = cv::imread(dirName + file_list[1], 0);

        // First rectify images
        cv::remap(original1, img1, mapX, mapY, cv::INTER_LINEAR);
        cv::remap(original2, img2, mapX, mapY, cv::INTER_LINEAR);

//        cv::imshow("First", img1);
//        cv::imshow("Second", img2);

#if 0
        //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        int minHessian = 400;

        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
        detector->setHessianThreshold(minHessian);

        std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
        cv::Mat descriptors_1, descriptors_2;
        detector->detectAndCompute(img1, cv::Mat(), keypoints_1, descriptors_1 );
        detector->detectAndCompute(img2, cv::Mat(), keypoints_2, descriptors_2 );

        //-- Step 2: Matching descriptor vectors using FLANN matcher
        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors_1, descriptors_2, matches);

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ ) {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < descriptors_1.rows; i++) {
            if (matches[i].distance <= std::max(2*min_dist, 0.02))  {
                good_matches.push_back(matches[i]);
            }
        }

        //-- Draw only "good" matches
        cv::Mat result;
        drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, result);
        cv::imshow("Result", result);
#endif

#if 0
        std::vector<cv::KeyPoint> kpts1, kpts2;
        cv::Mat desc1, desc2;
        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

        akaze->detectAndCompute(img1, cv::noArray(), kpts1, desc1);
        akaze->detectAndCompute(img2, cv::noArray(), kpts2, desc2);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> nn_matches;

        const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

        matcher.knnMatch(desc1, desc2, nn_matches, 2);
        std::vector<cv::KeyPoint> matched1, matched2;
        std::vector<cv::DMatch> good_matches;

        for (size_t i = 0; i < nn_matches.size(); i++) {
            cv::DMatch first = nn_matches[i][0];
            float dist1 = nn_matches[i][0].distance;
            float dist2 = nn_matches[i][1].distance;
            if (dist1 < nn_match_ratio * dist2) {
                int new_i = static_cast<int>(matched1.size());
                matched1.push_back(kpts1[first.queryIdx]);
                matched2.push_back(kpts2[first.trainIdx]);
                good_matches.push_back(cv::DMatch(new_i, new_i, 0));
            }
        }

        cv::Mat result;
        cv::drawMatches(img1, matched1, img2, matched2, good_matches, result);
        cv::imshow("Result", result);

#endif

#if 0
        cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 17);
        cv::Mat disparity, result;
        stereo->compute(img2, img1, disparity);
        cv::ximgproc::getDisparityVis(disparity, result);
        cv::imshow("Disparity", result);
#endif

#if 0
        cv::Mat flow, result;
        cv::calcOpticalFlowFarneback(img1, img2, flow, 0.4, 1, 12, 2, 8, 1.2, 0);
        cv::cvtColor(img1, result, cv::COLOR_GRAY2BGR);
        for (int y = 0; y < result.rows; y += 5) {
            for (int x = 0; x < result.cols; x += 5) {
                const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x) * 10;
                cv::line(result, cv::Point(x, y),
                    cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
                cv::circle(result, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
            }
        }
        cv::imshow("Flow", result);
#endif

#if 0
        cv::Mat img22;
        fakeMovement(params, params2, original1, img22, rotation, {-0.089, 0, 0});
        cv::Mat result = img22 - img2;
        cv::imshow("Source", img1);
        cv::imshow("Actual", img2);
        cv::imshow("Fake", img22);
        cv::imshow("Difference", result);
#endif

#if 0
        cv::Mat result;
        using namespace cv::ximgproc::segmentation;
        cv::Ptr<GraphSegmentation> seg = createGraphSegmentation();
        seg->processImage(img1, result);
        double minVal, maxVal;

        cv::minMaxLoc(result, &minVal, &maxVal);
        std::cout << "MIN " << minVal << " MAX " << maxVal << std::endl;
        cv::imshow("Result", result);

        cv::Mat colored;
        colored.create(result.rows, result.cols, CV_8UC3);

        for (int y = 0; y < result.rows; y += 5) {
            for (int x = 0; x < result.cols; x += 5) {
                int index = result.at<int>(y, x);
                cv::Scalar hsv = { index, 255, 255 };
                cv::Scalar bgr;
                cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
                colored.at<cv::Scalar>(y, x) = bgr;
            }
        }
        cv::imshow("Colored", colored);
#endif

        cv::Mat result;

        std::vector<cv::Point2f> previous;
        std::vector<cv::Point2f> current;
        auto criteria = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
        auto win_size = cv::Size(10, 10);

        cv::goodFeaturesToTrack(img1, previous, 1500, 0.01, 10);
        cv::cornerSubPix(img1, previous, win_size, cv::Size(-1, -1), criteria);

        std::vector<uint8_t> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(img1, img2, previous, current, status, err);

        std::cout << "PREVIOUS POINTS " << previous.size() << std::endl;

        cv::Mat source;
        cv::cvtColor(img1, source, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img2, result, cv::COLOR_GRAY2BGR);

        cv::Mat height(img2.rows, img2.cols, CV_8UC1, cv::Scalar(0));

        // Leave only valid points
        std::size_t i, k;
        double min_z = 100.0;
        double max_z = 0.0;

        for (i = k = 0; i < current.size(); i++) {
            if (!status[i]) {
                continue;
            }

            cv::circle(source, previous[i], 3, cv::Scalar(0, 255, 0), -1, 8);
            cv::circle(result, current[i], 3, cv::Scalar(0, 255, 0), -1, 8);
            previous[k] = previous[i];
            current[k] = current[i];
            k += 1;

            double distance = cv::norm(current[k] - previous[k]);
            double z = distance;
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
            if ( z > 20.0) z = 20.0;
            int color = 255 * (z / 20.0);
            cv::circle(height, current[i], 3, cv::Scalar(color), -1, 8);
        }

        previous.resize(k);
        current.resize(k);

        std::cout << "MIN " << min_z << " MAX " << max_z << std::endl;

        std::cout << "CURRENT POINTS " << current.size() << std::endl;

        cv::imshow("Source", source);
        cv::imshow("Result", result);
        cv::applyColorMap(height, height, cv::COLORMAP_HOT);
        cv::imshow("Height", height);

        cv::waitKey();
//    }

    return 0;
}

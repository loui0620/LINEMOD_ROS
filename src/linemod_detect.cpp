/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THISINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUS WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <ecto/ecto.hpp>
#include <fstream>
#include <iostream>
#include <math.h>

#include <boost/foreach.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#if CV_MAJOR_VERSION == 3
#include <opencv2/rgbd.hpp>
#else
#include <opencv2/rgbd/rgbd.hpp>
#endif

#include <object_recognition_core/db/ModelReader.h>
#include <object_recognition_core/common/pose_result.h>

#include "db_linemod.h"

#include <object_recognition_renderer/utils.h>
#include <object_recognition_renderer/renderer3d.h>

#include "linemod_icp.h"

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectDbPtr;

#include <opencv2/highgui/highgui.hpp>

#include "ros/ros.h"
#include "linemod_pointcloud.h"

// PCL dependencies
#include "PCL_ICP.h"
#include <pcl/point_types.h>

// #define DEBUG

LinemodPointcloud *pci_real_icpin_model;
LinemodPointcloud *pci_real_icpin_ref;
transformPublisher *transform_publisher;
objectIDPublisher *objID_publisher;
framePublisher *frame_publisher;
frameListener *listener;
sensorListener *sensorSub;
extICP *exticp;
poseCalculator *calculator;
tfListener *pose_listener;
requestListener *req_listener;
PositionArrListener *table_arr_listener;
//tfBroadcaster *pose_broadcaster;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

typedef Eigen::Affine3d Transformation;
typedef Eigen::Vector3d Point;
typedef Eigen::Vector3d Vector;
typedef Eigen::Translation<double, 3>  Translation;

std::map<std::string, float> global_boxHeightDict;
std::map<std::string, float> global_falseScoreDict;
std::map<std::string, std::string> global_modelPath;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

Transformation findTransformBetween2CS(Point fr0, Point fr1, Point fr2, Point to0, Point to1, Point to2) {

    // Define matrices and points :
    Eigen::Transform<double, 3, Eigen::Affine> T, T1, T2 = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
    Eigen::Matrix<double, 3, 1> x1, y1, x2, y2;
 
    // Axes of the coordinate system "fr"  
    x1 = (fr1 - fr0).normalized(); // the versor (unitary vector) of the (fr1-fr0) axis vector
    y1 = (fr2 - fr0).normalized();
 
    // Axes of the coordinate system "to"
    x2 = (to1 - to0).normalized();
    y2 = (to2 - to0).normalized();
 
    // transform from CS1 to CS2 
    // Note: if fr0==(0,0,0) --> CS1==CS2 --> T2=Identity
    T1.linear() << x1, y1, x1.cross(y1);
 
    // transform from CS1 to CS3
    T2.linear() << x2, y2, x2.cross(y2);
 
    // T = transform to CS2 to CS3
    // Note: if CS1==CS2 --> T = T3
    // T.linear() = T2.linear() * T1.linear().inverse(); 
 
    // T.translation() = to0;
    T.linear() = T2.linear() * T1.linear().inverse();
    // T.translation() = to0;
    T.translation() = to0 - (T.linear() * fr0);
    return T;
}

void parseMatxResult(const Eigen::Matrix4d &matrix, cv::Matx33d &R_ret, cv::Vec3d &T_ret)
{
	for(int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			R_ret(i, j) = (double)matrix(i, j);
		}
	}
	T_ret(0) = (double)matrix(0, 3);
	T_ret(1) = (double)matrix(1, 3);
	T_ret(2) = (double)matrix(2, 3);
}

Eigen::Matrix4d inverseTransform(Eigen::Matrix4d &mtx_in)
{
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.block(0, 0, 3, 3) = mtx_in.block(0, 0, 3, 3).transpose(); // R = R.inv()

    ret(0, 3) = mtx_in(0, 3) * -1;
    ret(1, 3) = mtx_in(1, 3) * -1;
    ret(2, 3) = mtx_in(2, 3) * -1;

    return ret;
}

Eigen::Matrix4d concatPose(Eigen::Matrix4d &mtx_global, Eigen::Matrix4d &mtx_local)
{
    Eigen::Matrix4d ret, global_inverse, local_inverse;
    Eigen::Affine3d aff_global, aff_local, aff_ret;
    Eigen::Quaterniond q_global, q_local, q_ret;

    //global_inverse = inverseTransform(mtx_global);
    global_inverse = mtx_global;
    aff_global.matrix() = global_inverse;
    aff_local.matrix() = mtx_local;
    
    aff_ret = aff_global * aff_local;
    ret = aff_ret.matrix();
    return ret;
    /*
    Eigen::Matrix4d ret;
    Eigen::Matrix3d rot_global, rot_local, rot_ret;
    
    rot_global = mtx_global.block(0, 0, 3, 3);
    rot_local = mtx_local.block(0, 0, 3, 3);
    
    Eigen::Vector3d tra_global(mtx_global(0, 3), mtx_global(1, 3), mtx_global(2, 3));
    Eigen::Vector3d tra_local(mtx_local(0, 3), mtx_local(1, 3), mtx_local(2, 3));
    
    Eigen::Vector3d tra_ret = tra_global + tra_local;
    rot_ret = rot_global * rot_local;

    ret.block(0, 0, 3, 3) = rot_ret;
    ret(0, 3) = tra_ret(0);
    ret(1, 3) = tra_ret(1);
    ret(2, 3) = tra_ret(2);
    ret(3, 3) = 1;

    return ret;
    */
}

cv::Mat equalizeIntensity(const cv::Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        cv::Mat ycrcb;

        cv::cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb,channels);

        cv::equalizeHist(channels[0], channels[0]);

        cv::Mat result;
        cv::merge(channels,ycrcb);
        cv::cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return cv::Mat();
}

void ImgSharpen(cv::Mat &data_in, cv::Mat &data_out)
{
  if (data_in.cols != 0 && data_in.rows != 0)
  {
    cv::Mat kern = (cv::Mat_<char>(3,3) << 0, -1 ,0,
                                   -1, 5, -1,
                                   0, -1, 0);
    filter2D(data_in, data_out, data_in.depth(), kern);
  } 
}

void CannyThreshold(cv::Mat &data_in, cv::Mat &data_out, cv::Rect &box)
{
  int edgeThresh = 1;
  int lowThreshold = 200;
  int ratio = 2;
  int kernel_size = 3;
  cv::Mat src_gray, detected_edges;
  cv::Rect rec;
  
  if (data_in.channels() == 3)
  {
    cv::cvtColor(data_in, src_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    // Convert 16UC1 to 8UC1
    cv::convertScaleAbs(data_in, data_in, 1, 0);
    src_gray = data_in;
  }

  /// Reduce noise with a kernel 3x3
  cv::blur( src_gray, detected_edges, cv::Size(3,3));
  
  /// Canny detector
  cv::Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
  /*
  rec = box;
  if(box.x != 0 && box.y != 0)
  {
    rec.x -= 10;
    rec.y -= 10;
    rec.width += 20;
    rec.height += 20;
  }
  detected_edges = detected_edges(rec);
  if (data_in.channels() != 1)
    cv::cvtColor(detected_edges, detected_edges, cv::COLOR_BGR2GRAY);
  */
  data_out = detected_edges;
 }

void appendPixelToVector(cv::Mat &data_in, std::vector<cv::Vec3f> &data_out, cv::Vec3f &center, float thres)
{
  for (int i = 0; i < data_in.cols; i++) {
    for (int j = 0; j < data_in.rows; j++) {
      int pixel = (int)data_in.at<uchar>(j, i);
      cv::Vec3f pts;
      if (pixel == 255)
      {
        pts(0) = i;
        pts(1) = j;
        pts(2) = sqrt((i - center(0)) * (i - center(0)) + (j - center(1)) * (j - center(1)));
        if (pts(2) < thres)
          data_out.push_back(pts);
      }
    }
  }
}

float CannyChecker(cv::Mat &color, cv::Mat &depth, cv::Rect &box)
{
  bool pointwise_debug = false;
  //prepare the bounding box for the model and reference point clouds
  cv::Rect_<int> rect_color(0, 0, color.cols, color.rows);
  //prepare the bounding box for the reference point cloud: add the offset
  cv::Rect_<int> rect_depth(0, 0, depth.cols, depth.rows);
  
  float score = -1.0;
  double alpha, beta;

  struct by_norm { 
    bool operator()(cv::Vec3f const &a, cv::Vec3f const &b) const { 
        return a(2) < b(2);
    }
  };
  
  if ((rect_color.width > 5) || (rect_color.height > 5))
  {
    //adjust both rectangles to be equal to the smallest among them
    if (rect_depth.width > rect_color.width)
      rect_depth.width = rect_color.width;
    if (rect_depth.height > rect_color.height)
      rect_depth.height = rect_color.height;
    if (rect_color.width > rect_depth.width)
      rect_color.width = rect_depth.width;
    if (rect_color.height > rect_depth.height)
      rect_color.height = rect_depth.height;

    cv::Mat color_out, depth_out, overlay_out;
    cv::Vec3f center(box.width / 2, box.height / 2, 0.0);
    // Vec3f: (pixel_x, pixel_y, norm)
    std::vector<cv::Vec3f> color_vec, depth_vec;

    CannyThreshold(color, color_out, box);
    CannyThreshold(depth, depth_out, box);
    
    color_out = color_out(rect_color);
    depth_out = depth_out(rect_depth);

    //std::cout << "Color Rect Size: " << color_out.size() << std::endl;
    //std::cout << "Depth Rect Size: " << depth_out.size() << std::endl;

    if (depth_out.size() == color_out.size())
    {
      appendPixelToVector(color_out, color_vec, center, 70.0);
      appendPixelToVector(depth_out, depth_vec, center, 70.0);

      std::sort(color_vec.begin(), color_vec.end(), by_norm());
      std::sort(depth_vec.begin(), depth_vec.end(), by_norm());
    }

    if (color_vec.size() != 0)
    {
      score = 0.0;
      int iter_max = std::min(color_vec.size(), depth_vec.size());
      
      cv::Mat_<float> features(0,2);
      std::vector<cv::Point2f> pointsForSearch;
      
      for(auto && point : color_vec) {
        
        //Fill matrix
        cv::Point2f pts = cv::Point2f(point(0), point(1));
        pointsForSearch.push_back(pts);
        
        //Fill matrix
        cv::Mat row = (cv::Mat_<float>(1, 2) << point(0), point(1));
        features.push_back(row);
      }
      
      cv::Mat source = cv::Mat(pointsForSearch).reshape(1);
      source.convertTo(source, CV_32F);
      
      int iterMax = std::min(pointsForSearch.size(), depth_vec.size());
      int invalidCount = 0;

      if (!pointsForSearch.empty())
      {
        for (int i = 0; i < iterMax; i++)
        {
          // Create an empty Mat for the features that includes dimensional
          // space for an x and y coordinate
          
          //cv::flann::Index flann_index(features, cv::flann::KDTreeIndexParams(1));
          cv::flann::Index flann_index(source, cv::flann::KDTreeIndexParams(2));

          unsigned int max_neighbours = 3;
          cv::Vec3f dValue = depth_vec[i];
          cv::Mat query = (cv::Mat_<float>(1, 2) << dValue(0), dValue(1));
          cv::Mat indices, dists; //neither assume type nor size here ! INDICES: 32SC1(int), DISTS: 32FC1(float)
          double radius= 1200.0;

          flann_index.knnSearch(query, indices, dists, max_neighbours, cv::flann::SearchParams(32));

          // If the closest distance > 5, consider as an invalid point and take out of score.
          if (sqrt(dists.at<float>(0, 0)) >= 5){
              invalidCount += 1;
              continue;
          }

          for (int j = 0; j < (int)max_neighbours; j++)
          {
            //False KNN-calculation checking
            if (indices.at<int>(j, 0) > pointsForSearch.size())
            {
              indices.at<int>(j, 0) = pointsForSearch.size();
            }

            //Calculate distance-score pixel-wisely.
            if (indices.at<int>(j, 0) < pointsForSearch.size() && indices.at<int>(j, 0) > -1)
            {
              if (dists.at<float>(j, 0) == 0) score += 100;
              else if (sqrt(dists.at<float>(j, 0)) == 1) score += 50;
              else if (sqrt(dists.at<float>(j, 0)) >  1 && sqrt(dists.at<float>(j, 0)) < 2) score += 20;
              else if (sqrt(dists.at<float>(j, 0)) >= 2 && sqrt(dists.at<float>(j, 0)) < 3) score += 5;
              else if (sqrt(dists.at<float>(j, 0)) >= 3 && sqrt(dists.at<float>(j, 0)) < 4) score += 2;
              else if (sqrt(dists.at<float>(j, 0)) >= 4 && sqrt(dists.at<float>(j, 0)) < 7) score += 1;
            }
          }
          if (pointwise_debug == true && !indices.empty())
          {
            std::cerr << "Indices 1, 2: " << indices.at<int>(0, 0) << ", " << indices.at<int>(1, 0) << std::endl;
            std::cerr << "Point 0: " << dValue << std::endl;
            std::cerr << "Nearest Point: " << pointsForSearch[indices.at<int>(0, 0)] << std::endl;
            std::cerr << "Indices: " << indices << std::endl << "Dists: " << dists << std::endl;
            std::cerr <<std::endl << std::endl;
          }
        }
        std::cout << "totalScore: " << score << std::endl;
        std::cout << "totalCount: " << iterMax << std::endl;
        std::cout << "invalidCount: " << invalidCount << std::endl;
        score /= (float)(iterMax - invalidCount);
      }
      
      alpha = 0.35;
      beta = ( 1.0 - alpha );
      cv::addWeighted( color_out, alpha, depth_out, beta, 0.0, overlay_out );
      //cv::imwrite("/home/simon/forDebug/canny_color.png", color_out);
      //cv::imwrite("/home/simon/forDebug/canny_overlay.png", overlay_out);
    }
    
  }
  return score;
}

void ptsFrom3DToDepth(cv::Vec3d &T_match, int centroid_x, int centroid_y, cv::Mat &K)
{
  float fx = K.at<float>(0, 0);
  float fy = K.at<float>(1, 1);
  float cx = K.at<float>(0, 2);
  float cy = K.at<float>(1, 2);
  centroid_x = (T_match(0) * fx / T_match(2)) + cx;
  centroid_y = (T_match(1) * fy / T_match(2)) + cy; 
}

void fromDepthTo3DMat(cv::Mat &depth, cv::Mat &K, cv::Mat_<cv::Vec3f> &out)
{
    int h = depth.rows;
    int w = depth.cols;
    
    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);

    if (h < 240){
      cx = (float)w * 0.5;
      cy = (float)h * 0.5;
    }
    //out = cv::Mat_<cv::Vec3f>(w, h);
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {

            unsigned short pixel = depth.at<ushort>(j, i);
            float z = (float)(pixel);
            z *= 0.001;
            if (z > 0){
              float xk = (float)((i - cx) / fx);
              float yk = (float)((j - cy) / fy);
              out(j, i)[0] = xk * z;
              out(j, i)[1] = yk * z;
              out(j, i)[2] = z;
              //if (i == 320 && j == 240) std::cout << "zVal: " << z << ", ---";
            }  
        }
    }
}

void fromDepthTo3D_FloatFormat(cv::Mat &depth, std::vector<cv::Vec3f> &out, cv::Mat K)
{
    // For another subscriber at CV_32FC1 format
    int h = depth.rows;
    int w = depth.cols;
    cv::Mat tmp;
    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);
    
    if (h < 240){
      cx = (float)w * 0.5;
      cy = (float)h * 0.5;
    }
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            cv::Vec3f vec;
            
            float z = depth.at<float>(j, i);
            //z *= 0.001;
            if (z > 0){
              float xk = (float)((i - cx) / fx);
              float yk = (float)((j - cy) / fy);
              vec(0) = xk * z;
              vec(1) = yk * z;
              vec(2) = z;
              //out.at<cv::Vec3f>(j, i) = vec;
              //std::cout << "pts: " << vec << std::endl;
              out.push_back(vec);
            }  
        }
    }
}

void fromDepthTo3D(cv::Mat &depth, std::vector<cv::Vec3f> &out, cv::Mat K)
{
    int h = depth.rows;
    int w = depth.cols;
    cv::Mat tmp;
    float fx = K.at<float>(0, 0);
    float fy = K.at<float>(1, 1);
    float cx = K.at<float>(0, 2);
    float cy = K.at<float>(1, 2);

    if (h < 240){
      cx = (float)w * 0.5;
      cy = (float)h * 0.5;
    }
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            cv::Vec3f vec;
            unsigned short pixel = depth.at<ushort>(j, i);
            float z = (float)(pixel);
            z *= 0.001;
            if (z > 0){
              float xk = (float)((i - cx) / fx);
              float yk = (float)((j - cy) / fy);
              vec(0) = xk * z;
              vec(1) = yk * z;
              vec(2) = z;
              //out.at<cv::Vec3f>(j, i) = vec;
              //std::cout << "pts: " << vec << std::endl;
              out.push_back(vec);
            }  
        }
    }
}
float distance(cv::Point3f &a, cv::Point3f &b) {
  return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z));
}

void fillPixel(cv::Mat &blank, cv::Mat &boundBox, int offSetX, int offSetY)
{
  //std::cout << "matchXY: " << offSetX << ", " << offSetY << std::endl;
  int h = boundBox.rows;
  int w = boundBox.cols;
  //printf("boxSize: %i, %i. ", w, h);
  for (int i = 0; i < w; i++){
    for (int j = 0; j < h; j++)
      blank.at<ushort>(j + offSetY, i + offSetX) = boundBox.at<ushort>(j, i);
  }
}

void
drawResponse(const std::vector<cv::linemod::Template>& templates, int num_modalities, cv::Mat& dst, cv::Point offset,
             int T)
{
  static const cv::Scalar COLORS[5] =
  { CV_RGB(0, 0, 255), CV_RGB(0, 255, 0), CV_RGB(255, 255, 0), CV_RGB(255, 140, 0), CV_RGB(255, 0, 0) };
  if (dst.channels() == 1)
    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

// Red Dot as Position Reference.
  //cv::circle(dst, cv::Point(offset.x + 20, offset.y + 20), T / 2, COLORS[4]);
  if (num_modalities > 5)
    num_modalities = 5;
  
  float cent_x, cent_y;
  for (int m = 0; m < num_modalities; ++m)
  {
// NOTE: Original demo recalculated max response for each feature in the TxT
// box around it and chose the display color based on that response. Here
// the display color just depends on the modality.
    cv::Scalar color = COLORS[m];
    
    int x_sum = 0;
    int y_sum = 0;

    for (int i = 0; i < (int) templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
      x_sum += f.x;
      y_sum += f.y;
    }
    cent_x = (float)x_sum / (float)templates[m].features.size();
    cent_y = (float)y_sum / (float)templates[m].features.size(); 
  }
  // draw CENTROID
  cv::circle(dst, cv::Point((int)cent_x + offset.x, (int)cent_y + offset.y), T, COLORS[4]); 
  std::string saveName = "";
  //cv::imwrite(saveName, dst);
}

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

namespace ecto_linemod
{
struct Detector: public object_recognition_core::db::bases::ModelReaderBase {
  void parameter_callback(
      const object_recognition_core::db::Documents & db_documents) {
    /*if (submethod.get_str() == "DefaultLINEMOD")
     detector_ = cv::linemod::getDefaultLINEMOD();
     else
     throw std::runtime_error("Unsupported method. Supported ones are: DefaultLINEMOD");*/
    std::string pointcloud_path;
    if(!(*use_rgb_) && !(*use_depth_))
      throw std::runtime_error("Unsupported type of input data: either use_rgb or use_depth (or both) parameters shouled be true");
    if(!(*use_rgb_) && *use_depth_)
      std::cout << "WARNING:: Gradients computation will be based on depth data (but not rgb image)." << std::endl;
    detector_ = cv::linemod::getDefaultLINEMOD();
    
    BOOST_FOREACH(const object_recognition_core::db::Document & document, db_documents) {
      std::string object_id = document.get_field<ObjectId>("object_id");
      
      // Add JSON parser here
      Json::Value Root;
      Json::Reader jsonReader;

      std::fstream ifs(*json_path_);
      if(!ifs)
      {
        ROS_INFO("JSON: NO JSON FILE EXISTED.");
      }
      if(!jsonReader.parse(ifs, Root)){
        ROS_INFO("JSON: FAIL TO LOAD DATA FROM JSON");
      }
      
      std::cout << "Read: " << Root[object_id]["name"].asString() << std::endl;
      std::cout << "PLY model: " << Root[object_id]["model_path"].asString() << std::endl;
      std::string model_path_foreach = Root[object_id]["model_path"].asString();
      
      //OBJECT_HEIGHT = Root[object_id]["box_height"].asFloat();
      //FALSE_CHECK_SCORE = Root[object_id]["false_check_score"].asFloat();
      
      global_boxHeightDict[object_id] = Root[object_id]["box_height"].asFloat();
      global_falseScoreDict[object_id] = Root[object_id]["false_check_score"].asFloat();
      global_modelPath[object_id] = Root[object_id]["model_path"].asString();

      // Add Mesh File
      exticp = new extICP();
      exticp->loadModel(model_path_foreach, object_id);
      exticp->previous_rect.x = 0;
      exticp->previous_rect.y = 0;
      exticp->previous_rect.width = 20;
      exticp->previous_rect.height = 20;

      // Load the detector for that class
      cv::linemod::Detector detector;
      document.get_attachment<cv::linemod::Detector>("detector", detector);
      if (detector.classIds().empty())
        continue;
      std::string object_id_in_db = detector.classIds()[0];
      for (size_t template_id = 0; template_id < detector.numTemplates();
          ++template_id) {
        const std::vector<cv::linemod::Template> &templates_original = detector.getTemplates(object_id_in_db, template_id);
        detector_->addSyntheticTemplate(templates_original, object_id);
      }

      // Deal with the poses
      document.get_attachment<std::vector<cv::Mat> >("Rs", Rs_[object_id]);
      document.get_attachment<std::vector<cv::Mat> >("Ts", Ts_[object_id]);
      document.get_attachment<std::vector<float> >("distances", distances_[object_id]);
      document.get_attachment<std::vector<cv::Mat> >("Ks", Ks_[object_id]);
      renderer_n_points_ = document.get_field<int>("renderer_n_points");
      renderer_angle_step_ = document.get_field<int>("renderer_angle_step");
      renderer_radius_min_  = document.get_field<double>("renderer_radius_min");
      renderer_radius_max_ = document.get_field<double>("renderer_radius_max");
      renderer_radius_step_ = document.get_field<double>("renderer_radius_step");
      renderer_width_ = document.get_field<int>("renderer_width");
      renderer_height_ = document.get_field<int>("renderer_height");
      renderer_focal_length_x_ = document.get_field<double>("renderer_focal_length_x");
      renderer_focal_length_y_ = document.get_field<double>("renderer_focal_length_y");
      renderer_near_ = document.get_field<double>("renderer_near");
      renderer_far_ = document.get_field<double>("renderer_far");

      if (setupRenderer(object_id))
        std::cout << "Loaded " << object_id
                << " with the number of samples " << Rs_[object_id].size() << std::endl << std::endl;
    }
    
    //initialize the visualization

    ros::NodeHandle node_;
    pci_real_icpin_model = new LinemodPointcloud(node_, "real_icpin_model", *depth_frame_id_); //GREEN PCs
    pci_real_icpin_ref = new LinemodPointcloud(node_, "real_icpin_ref", *depth_frame_id_); // BLUE PCs
    transform_publisher = new transformPublisher(node_, "linemod_response", *depth_frame_id_); // Transform Matrix Publisher
    objID_publisher = new objectIDPublisher(node_, "linemod_response_id", *depth_frame_id_);
    frame_publisher = new framePublisher(node_, "camera/draw_response", *depth_frame_id_); // Publish linemod-detection images
    listener = new frameListener(node_, "/camera/rgb/image_raw","/camera/depth/image_raw", *depth_frame_id_);
    sensorSub = new sensorListener(node_, "/sensor_distance", *depth_frame_id_);
    req_listener = new requestListener(node_, "/linemod_request", *depth_frame_id_);
    calculator = new poseCalculator;
    pose_listener = new tfListener;
    table_arr_listener = new PositionArrListener(node_, "/tabletop_response", *depth_frame_id_);
    //pose_broadcaster = new tfBroadcaster;

  }

    static void
    declare_params(tendrils& params)
    {
      object_recognition_core::db::bases::declare_params_impl(params, "LINEMOD");
      params.declare(&Detector::threshold_, "threshold", "Matching threshold, as a percentage", 89.0f);
      params.declare(&Detector::visualize_, "visualize", "If True, visualize the output.", true);
      params.declare(&Detector::use_rgb_, "use_rgb", "If True, use rgb-based detector.", true);
      params.declare(&Detector::use_depth_, "use_depth", "If True, use depth-based detector.", true);
      params.declare(&Detector::th_obj_dist_, "th_obj_dist", "Threshold on minimal distance between detected objects.", 0.04f); // 0.04f
      params.declare(&Detector::verbose_, "verbose", "If True, print.", false);
      params.declare(&Detector::depth_frame_id_, "depth_frame_id", "The depth camera frame id.", "camera_depth_optical_frame");
      params.declare(&Detector::icp_dist_min_, "icp_dist_min", "", 0.06f); //0.06f
      params.declare(&Detector::px_match_min_, "px_match_min", "", 0.25f);
      params.declare(&Detector::json_path_, "json_path", "", "");
      params.declare(&Detector::false_positive_threshold_, "false_positive_threshold", "", 1400.0);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare(&Detector::color_, "image", "An rgb full frame image.");
      inputs.declare(&Detector::depth_, "depth", "The 16bit depth image.");
      inputs.declare(&Detector::K_depth_, "K_depth", "The calibration matrix").required();

      outputs.declare(&Detector::pose_results_, "pose_results", "The results of object recognition");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      configure_impl();
    }

    /**
     * @brief Initializes the renderer with the parameters used in the training phase.
     * The renderer will be later used to render depth clouds for each detected object.
     * @param[in] The object id to initialize.*/
    bool
    setupRenderer(const std::string &object_id)
    {
      object_recognition_core::db::ObjectDbParameters db_params(*json_db_);
      // Get the document for the object_id_ from the DB
      object_recognition_core::db::ObjectDbPtr db = db_params.generateDb();
      object_recognition_core::db::Documents documents =
          object_recognition_core::db::ModelDocuments(db,
              std::vector<object_recognition_core::db::ObjectId>(1,
                  object_id), "mesh");
      if (documents.empty()) {
        std::cerr << "Skipping object id \"" << object_id
            << "\" : no mesh in the DB" << std::endl;
        return false;
      }

      // Get the list of _attachments and figure out the original one
      object_recognition_core::db::Document document = documents[0];
      std::vector<std::string> attachments_names = document.attachment_names();
      std::string mesh_path;
      std::vector<std::string> possible_names(2);
      possible_names[0] = "original";
      possible_names[1] = "mesh";
      for (size_t i = 0; i < possible_names.size() && mesh_path.empty(); ++i) {
        BOOST_FOREACH(const std::string& attachment_name, attachments_names){
          if (attachment_name.find(possible_names[i]) != 0)
            continue;
          std::cout << "Reading the mesh file " << attachment_name << std::endl;
          // Create a temporary file
          char mesh_path_tmp[L_tmpnam] = "/tmp/linemod_XXXXXX";
          mkstemp(mesh_path_tmp);
          mesh_path = std::string(mesh_path_tmp) + attachment_name.substr(possible_names[i].size());
          
          // Load the mesh and save it to the temporary file
          std::ofstream mesh_file;
          mesh_file.open(mesh_path.c_str());
          document.get_attachment_stream(attachment_name, mesh_file);
          mesh_file.close();
          std::string str = mesh_path.c_str();
  
        }
      }
      
      // the model name can be specified on the command line.
      Renderer3d *renderer_ = new Renderer3d(mesh_path);
      renderer_->set_parameters(renderer_width_, renderer_height_, renderer_focal_length_x_, renderer_focal_length_y_, renderer_near_, renderer_far_);

      // std::remove(mesh_path.c_str());

      //initiaization of the renderer with the same parameters as used for learning
      RendererIterator *renderer_iterator_ = new RendererIterator(renderer_, renderer_n_points_);
      renderer_iterator_->angle_step_ = renderer_angle_step_;
      renderer_iterator_->radius_min_ = float(renderer_radius_min_);
      renderer_iterator_->radius_max_ = float(renderer_radius_max_);
      renderer_iterator_->radius_step_ = float(renderer_radius_step_);
      renderer_iterators_.insert(std::pair<std::string,RendererIterator*>(object_id, renderer_iterator_));
      return true;
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      int count = 0;

#ifndef DEBUG
      if (!req_listener->m_isRequested) {
        if (table_arr_listener->positions_vec != NULL) {
         std::cout << "tabletop array size:" << table_arr_listener->positions_vec->size() << std::endl;
        }
          
        ros::Duration(2.0).sleep();
        ROS_INFO("LineMOD Waiting for request...");
        return ecto::OK;
      }
#endif
      /*
      while (!req_listener->m_isRequested) {
        if (count > 10) {
          ROS_INFO("Null request timeout, break.");
          break;
        }
        ros::Duration(2).sleep();
        ROS_INFO("Waiting for request...");
        count++;
      }
      */
      PoseResult pose_result;
      pose_results_->clear();

      if (detector_->classIds().empty())
        return ecto::OK;

      std::vector<cv::Mat> sources;

      // Resize color to 640x480
      /// @todo Move resizing to separate cell, and try LINE-MOD w/ SXGA images

      cv::Mat display, color_display;
      if (*use_rgb_)
      {
        cv::Mat color;
        if (color_->rows > 960)
          cv::pyrDown(color_->rowRange(0, 960), color);
        else
          color_->copyTo(color);
          if(!listener->color_from_listener.empty())
            listener->color_from_listener.copyTo(color); // Copy frame from external listener
          
          /*
          cv::imwrite("/home/louis/forDebug/color_src.png", color);
          std::string ty =  type2str( listener->color_from_listener.type() );
          printf("Color Listener: %s %dx%d \n", ty.c_str(), listener->color_from_listener.cols, listener->color_from_listener.rows );
          */

          //color_display = equalizeIntensity(color);
          //cv::GaussianBlur(color, color, cv::Size(3, 3), 0, 0);
          ImgSharpen(color, color_display);
          color = color_display;          
          
          exticp->stored_color.release();
          color(exticp->previous_rect).copyTo(exticp->stored_color);


        if (*visualize_)
          display = color;
          sources.push_back(color);
          frame_publisher->fillImageAndPublish(display);
      }
      
      cv::Mat depth = *depth_;
      //cv::Mat depth = listener->depth_from_listener.clone();
      cv::Mat depth_clipped;

      if (depth_->depth() == CV_32FC1){
        depth_->convertTo(depth, CV_16UC1, 1000.0);
      }
      
      if (*use_depth_) 
      {
        if (!(*use_rgb_))
        {
          //add a depth-based gray image to the list of sources for matching
          depth.convertTo(display, CV_8U);
          cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
          sources.push_back(display);
        }
        //cv::medianBlur(depth, depth, 3);
        sources.push_back(depth);
      }

      std::vector<cv::linemod::Match> matches;
      detector_->match(sources, *threshold_, matches);

      int num_modalities = (int) detector_->getModalities().size();

      cv::Mat_<cv::Vec3f> depth_real_ref_raw; // Size(640, 480)
      
      float query_data[9] = {601.444, 0, 320.3814, 0, 607.7046, 246.282, 0, 0, 1};   //astra pro
      //float query_data[9] = {619.444, 0, 320.3814, 0, 619.7046, 240.282, 0, 0, 1}; //realsense2
      cv::Mat K_dummy = cv::Mat(3, 3, CV_32F, query_data);
      
      cv::Mat_<float> K;
      K_depth_->convertTo(K, CV_32F);
      
      cv::depthTo3d(depth, K, depth_real_ref_raw);
      fromDepthTo3DMat(depth, K, depth_real_ref_raw);

      std::vector<cv::Vec3f> point_ref_temp;
      point_ref_temp.clear();
      //fromDepthTo3D(depth, point_ref_temp, K);

      // Apply previous ROI to depth src then fill into blank image.
      cv::Vec3d t_dummy = cv::Vec3d(0.0, 0.0, 0.0);
      depth(exticp->previous_rect).copyTo(depth_clipped);
      
      cv::Mat container = cv::Mat::zeros(cv::Size(640, 480), CV_16U);
      fillPixel(container, depth_clipped, exticp->previous_rect.x, exticp->previous_rect.y);
      fromDepthTo3D(container, point_ref_temp, K);
          
      PointCloudT::Ptr scene_cloud(new PointCloudT);
      
      pcl_Vector3fToPC(point_ref_temp, scene_cloud, 2);
      pcl_FilterByDistance(scene_cloud, scene_cloud, 3.0);

      pcl::gpu::Octree::PointCloud scene_cloud_gpu;
      pcl::gpu::Octree scene_tree_gpu; // remember move out
      scene_cloud_gpu.upload(scene_cloud->points);
      scene_tree_gpu.setCloud(scene_cloud_gpu);
      scene_tree_gpu.build();
      
      /*
      // TEMP DEBUG---------------------------------------------------------------
      cv::Mat depth_dummy = container;
      std::vector<cv::Vec3f> point_roi_ref;
      depth(exticp->previous_rect).copyTo(depth_dummy); //ROI-filter by previous model position.
      
      //cv::imwrite("/home/simon/forDebug/rected.png", depth_dummy);
      if(exticp->previous_rect.x != 0) std::cout << "\nSTORED Rect Value: \n" << exticp->previous_rect << std::endl;
      PointCloudT::Ptr m_cloud_debug(new PointCloudT);
      cv::Mat container = cv::Mat::zeros(cv::Size(640, 480), CV_16U);
      fillPixel(container, depth_dummy, exticp->previous_rect.x, exticp->previous_rect.y); // [out] blankImg
      
      fromDepthTo3D(container, point_roi_ref, K_dummy);
      pcl_Vector3fToPC(point_roi_ref, m_cloud_debug, 1, "/home/simon/forDebug/rected.png", t_dummy, false);
      std::string dbgName = "/home/simon/forDebug/model/ROI-clipped_ref.ply";
      if(m_cloud_debug->points.size() != 0) pcl::io::savePLYFileASCII(dbgName, *m_cloud_debug);

      // TEMP DEBUG---------------------------------------------------------------
      */
     
      int iter = 0;
      //clear the vector of detected objects
      objs_.clear();
      //clear the point clouds
#if LINEMOD_VIZ_PCD
      pci_real_icpin_model->clear();
      pci_real_icpin_ref->clear();
#endif
    
    if (matches.size() > 5) {
      std::vector<cv::linemod::Match> top_five_templates = {matches.begin(), matches.begin() + 5};
      matches = top_five_templates;
    }

    BOOST_FOREACH(const cv::linemod::Match & match, matches) {

#ifndef DEBUG
      if (req_listener->m_isRequested == false) continue;
#endif
      const std::vector<cv::linemod::Template>& templates =
          detector_->getTemplates(match.class_id, match.template_id);

      std::cout << "OBJECT ID: " << match.class_id << std::endl;
      //std::cout << "TEMPLATE ID: " << match.template_id << "-------------------------------------------------------------"  << std::endl;

      int idMatch = match.template_id;
      
      std::map<std::string, float>::iterator height_iter;
      std::map<std::string, float>::iterator score_iter;

      height_iter = global_boxHeightDict.find(match.class_id);
      score_iter = global_falseScoreDict.find(match.class_id);

      float OBJECT_HEIGHT = height_iter->second;
      float FALSE_CHECK_SCORE = score_iter->second;

      if (*visualize_)
        drawResponse(templates, num_modalities, display,
            cv::Point(match.x, match.y), detector_->getT(0));
      
      // Pass detection result to Rviz
      // frame_publisher->fillImageAndPublish(display);

      // Fill the Pose object
      cv::Matx33d R_match = Rs_.at(match.class_id)[match.template_id].clone();
      cv::Vec3d T_match = Ts_.at(match.class_id)[match.template_id].clone();
      float D_match = distances_.at(match.class_id)[match.template_id];
      cv::Mat K_match = Ks_.at(match.class_id)[match.template_id];

      //get the point cloud of the rendered object model  
      cv::Mat mask;
      cv::Rect rect;
      cv::Matx33d R_temp(R_match.inv());
      cv::Vec3d up(-R_temp(0,1), -R_temp(1,1), -R_temp(2,1));
      RendererIterator* it_r = renderer_iterators_.at(match.class_id);
      cv::Mat depth_ref_, color_ref_;
      it_r->render(color_ref_, depth_ref_, mask, rect, -T_match, up);

      cv::Mat blankImg = cv::Mat::zeros(cv::Size(640, 480), CV_16U);
      fillPixel(blankImg, depth_ref_, match.x, match.y); // [out] blankImg

      cv::Mat_<cv::Vec3f> depth_real_model_raw;
      
      K_match.at<float>(0, 0) = K.at<float>(0, 0);
      K_match.at<float>(1, 1) = K.at<float>(1, 1);
      cv::depthTo3d(depth_ref_, K_match, depth_real_model_raw);
      
      //std::cout << "R_match: " << std::endl << R_match << std::endl;
      //std::cout << "T_match: " << std::endl << T_match << std::endl;

      // TEMP OBJECT FOR PUBLISH
      std::vector<cv::Vec3f> point_model_temp;
      std::vector<cv::Vec3f> point_model_out;
      std::vector<cv::Vec3f> point_mesh_out;
      fromDepthTo3D(blankImg, point_model_temp, K);
      
      //prepare the bounding box for the model and reference point clouds
      cv::Rect_<int> rect_model(0, 0, depth_real_model_raw.cols, depth_real_model_raw.rows);
      //prepare the bounding box for the reference point cloud: add the offset
      cv::Rect_<int> rect_ref(rect_model);
      rect_ref.x += match.x;
      rect_ref.y += match.y;
      //std::cout << "RECT_MODEL: " << rect_ref << std::endl;

      rect_ref = rect_ref & cv::Rect(0, 0, depth_real_ref_raw.cols, depth_real_ref_raw.rows);
      //std::cout << "RECT_REF: " << rect_ref << std::endl;
      if ((rect_ref.width < 5) || (rect_ref.height < 5))
        continue;
      //adjust both rectangles to be equal to the smallest among them
      if (rect_ref.width > rect_model.width)
        rect_ref.width = rect_model.width;
      if (rect_ref.height > rect_model.height)
        rect_ref.height = rect_model.height;
      if (rect_model.width > rect_ref.width)
        rect_model.width = rect_ref.width;
      if (rect_model.height > rect_ref.height)
        rect_model.height = rect_ref.height;
      
      exticp->previous_rect = rect_ref;

      // Use Canny Thrshold to check initial result of alignment.
      float canny_score = 10.0;
      //float canny_score = CannyChecker(extip->stored_color, depth_ref_, exticp->previous_rect);
      std::cout << std::endl << "Canny Score: " << canny_score << "-------------------------------------------------------------"  << std::endl << std::endl;
      
      //if (canny_score < 15) continue;

      //prepare the reference data: from the sensor : crop images
      cv::Mat_<cv::Vec3f> depth_real_ref = depth_real_ref_raw(rect_ref);
      //prepare the model data: from the match
      cv::Mat_<cv::Vec3f> depth_real_model = depth_real_model_raw(rect_model);
      
      //get the point clouds (for both reference and model)
      PointCloudT::Ptr m_cloud_model(new PointCloudT);
      PointCloudT::Ptr m_cloud_ref(new PointCloudT);
      PointCloudT::Ptr m_cloud_icp(new PointCloudT);
      cv::Matx33d R_publish = R_match;
      cv::Vec3d T_publish = 1 * T_match;
      Eigen::Matrix4d trans_full;

      //TODO: investigate score tendencies in distant range
      float confidence_thres = *false_positive_threshold_;
      
      std::string saveIDs = std::to_string(match.template_id);
      std::string modelName = "/home/simon/forDebug/model/model-" + saveIDs + ".ply";
      std::string refName = "/home/simon/forDebug/ref/ref-" + saveIDs + ".ply";
      
      //pcl_Vector3fToPC(point_model_temp, m_cloud_model, 2, T_publish, false);
      //pcl_Vector3fToPC(point_ref_temp, m_cloud_ref, 2, T_publish, false);
      pcl_Vector3fToPC(point_model_temp, m_cloud_model, 2);
      pcl_Vector3fToPC(point_ref_temp, m_cloud_ref, 2);
      
      pcl_FilterByDistance(m_cloud_model, m_cloud_model, 3.0);
      pcl_FilterByDistance(m_cloud_ref, m_cloud_ref, 3.0);
      
      // Set YZ offset for better ICP result.
      float Y_pre = -0.00;
      float Z_pre = -0.02;
      
      Eigen::Matrix4d pre_transform_1 = Eigen::Matrix4d::Identity();
      pre_transform_1(1, 3) = Y_pre;
      pre_transform_1(2, 3) = Z_pre;
      
      pcl::transformPointCloud(*m_cloud_model, *m_cloud_model, pre_transform_1);

      // Run ICP algorithm with Template Model(Estimated Pose) and Input Source(Depth Image)
      float first_confidence_score = pcl_runICP(m_cloud_model, m_cloud_ref, m_cloud_icp, scene_tree_gpu, R_publish, T_publish, trans_full, 20, false);
      
      T_publish(1) -= Y_pre;
      T_publish(2) -= Z_pre;
      /* 
      if (first_confidence_score < first_confidence_thres)
      {
        std::string modelname = "/home/simon/forDebug/model/" + saveIDs + "-1st-model_before.ply";
        pcl::io::savePLYFileASCII(modelname, *m_cloud_model);
        std::string stlname = "/home/simon/forDebug/model/" + saveIDs + "-1st-reference.ply";
        pcl::io::savePLYFileASCII(stlname, *m_cloud_ref);
        std::string modelName = "/home/simon/forDebug/model/" + saveIDs + "-1st-model-final.ply";
        pcl::io::savePLYFileASCII(modelName, *m_cloud_icp);
      }
      */
      point_model_temp.clear();
      pcl_PCToVector3f(m_cloud_icp, point_model_out);
      
      //if ((first_confidence_score < confidence_thres))
      if ((first_confidence_score < FALSE_CHECK_SCORE))
      {
        pci_real_icpin_model->clear();
        pci_real_icpin_model->fill(point_model_out, cv::Vec3b(0,255,0));
        pci_real_icpin_model->publish();
      }
      
      cv::Vec3f T_model_out;
      pcl_computeCentroid(point_model_out, T_model_out);

      if ((first_confidence_score > FALSE_CHECK_SCORE) && (point_model_temp.size() < 1000)) 
        continue;

      // -------------------------------------------2nd Phase Computing-------------------------------------------
      //initialize the translation based on reference data
      cv::Vec3f T_crop = depth_real_ref((depth_real_ref.rows / 2.0f), depth_real_ref.cols / 2.0f); // As cv::Mat_ M(j, i)
      //add the object's depth
      T_crop(2) += D_match;
      std::cerr << "T_crop: " << std::endl << T_crop << std::endl << "D_match: " << D_match << std::endl;
      
      if (!cv::checkRange(T_crop))
        continue;
      cv::Vec3f T_real_icp(T_crop);

      //initialize the rotation based on model data
      if (!cv::checkRange(R_match))
        continue;
      cv::Matx33f R_real_icp(R_match);

      std::vector<cv::Vec3f> pts_real_model_temp;
      std::vector<cv::Vec3f> pts_real_ref_temp;

      // Pre-offset of stl-object
      cv::Matx31d centroid_vec = {0.0, 0.0, 0.0}; // Model Centroid As Post-Compensator.{-0.005, 0.0, -0.015}
      centroid_vec = 1 * R_publish * centroid_vec;
      T_model_out(0) += centroid_vec(0, 0);
      T_model_out(1) += centroid_vec(0, 1);
      T_model_out(2) += centroid_vec(0, 2);

      // Re-set intial guess of translation to centroid of POINT_MODEL_OUT
      T_crop(0) = T_model_out(0);
      T_crop(1) = T_model_out(1);
      //T_crop(2) += centroid_vec(0, 2);
      

      //get the point clouds (for both reference and model)
      PointCloudT::Ptr cloud_model(new PointCloudT);
      PointCloudT::Ptr cloud_ref(new PointCloudT);
      PointCloudT::Ptr cloud_icp(new PointCloudT);
      Eigen::Matrix4d trans_ork = Eigen::Matrix4d::Identity();
      cv::Matx33d R_ork;
      cv::Vec3d T_ork;

      // Read PC model from external class.
      Eigen::Matrix4d trans_tmp;
      pcl_bundleMatxResult(trans_tmp, R_publish, T_crop);
      
      std::map<std::string, std::string>::iterator path_iter;
      path_iter = global_modelPath.find(match.class_id);
      std::string modelPath = path_iter->second;
      //cloud_ref = exticp->m_mesh.makeShared();
      cloud_ref = exticp->loadSingleModel(modelPath);
      
      pcl::transformPointCloud(*cloud_ref, *cloud_ref, trans_tmp); // Based on lastly result, transform reference to the position.

      R_ork = R_publish;
      T_ork(0) = (double)T_crop(0);
      T_ork(1) = (double)T_crop(1);
      T_ork(2) = (double)T_crop(2);
      
      float px_ratio_missing = matToVec(depth_real_ref, depth_real_model, pts_real_ref_temp, pts_real_model_temp);

      cloud_model = m_cloud_icp;
      
      // pre-processing: compute centroid of two PC-set then align
      float z_pre = -0.02;
      Eigen::Vector4f centroid_in, centroid_target;
      Eigen::Matrix4d pre_transform = Eigen::Matrix4d::Identity();
      pcl::compute3DCentroid(*cloud_model, centroid_in);
      pcl::compute3DCentroid(*cloud_ref, centroid_target);
      
      // Set a good initial-pose then compensate the translation vector.
      pre_transform(0, 3) = centroid_target[0] - centroid_in[0];
      pre_transform(1, 3) = centroid_target[1] - centroid_in[1];
      pre_transform(2, 3) = z_pre;
      pcl::transformPointCloud(*cloud_model, *cloud_model, pre_transform);
      
      T_ork(0) -= (centroid_target[0] - centroid_in[0]);
      T_ork(1) -= (centroid_target[1] - centroid_in[1]);
      T_ork(2) -= z_pre;

      // Align mesh according to the init-result from first-ICP.
      float second_confidence_score = pcl_runICP(cloud_model, cloud_ref, cloud_icp, R_ork, T_ork, trans_ork, 60, true);
      point_model_temp.clear();
      
      pcl_PCToVector3f(cloud_icp, point_model_out);
      pcl_PCToVector3f(cloud_ref, point_ref_temp);
      
      if ((first_confidence_score > FALSE_CHECK_SCORE)) continue; 
      /*
      if (second_confidence_score < confidence_thres)
      {
        std::string modelname = "/home/simon/forDebug/model/" + saveIDs + "-model_before.ply";
        pcl::io::savePLYFileASCII(modelname, *cloud_model);
        std::string stlname = "/home/simon/forDebug/model/" + saveIDs + "-transformedSTL.ply";
        pcl::io::savePLYFileASCII(stlname, *cloud_ref);
        std::string modelName = "/home/simon/forDebug/model/" + saveIDs + "-model-final.ply";
        pcl::io::savePLYFileASCII(modelName, *cloud_icp);
      }
      */
      //float px_ratio_missing = matToVec(depth_real_ref, depth_real_model, pts_real_ref_temp, pts_real_model_temp);
      if (px_ratio_missing > (1.0f-*px_match_min_))
        continue;
      
      //perform the first approximate ICP
      float px_ratio_match_inliers = 0.0f;
      float icp_dist = 0.03f;
      //float icp_dist = icpCloudToCloud(pts_real_ref_temp, pts_real_model_temp, R_real_icp, T_real_icp, px_ratio_match_inliers, 1);

      //reject the match if the icp distance is too big
      if (icp_dist > *icp_dist_min_)
        continue;
      
      pcl_bundleMatxResult(trans_ork, R_publish, T_crop);
      Eigen::Matrix4d trans_meaned = Eigen::Matrix4d::Identity();
      Eigen::Matrix4d camera_pose = Eigen::Matrix4d::Identity(), table_pose, concated;
      
      /**           GET TABLE COORDINATE AND ALIGN OBJECT ON IT.          **/
      
      //pose_listener->getRelativePose(camera_pose, "/camera_link");
      pose_listener->getRelativePose(camera_pose, "/camera_depth_optical_frame");
      pose_listener->getRelativePose(table_pose, "/table_pose");

      concated = concatPose(camera_pose, trans_ork);

      Eigen::Vector3d origin(0, 0, 0), unit_x(1, 0, 0), unit_y(0, 1, 0), unit_z(0, 0, 1);
      Eigen::Vector3d obj_unit_x, obj_unit_z, table_unit_x, table_unit_z, tmp_translation;
      
      tmp_translation = Eigen::Vector3d(concated(0, 3), concated(1, 3), concated(2, 3));

      obj_unit_x = concated.block(0, 0, 3, 3) * unit_x;
      obj_unit_z = concated.block(0, 0, 3, 3) * unit_y; // Y as vertical axis in STL file
      table_unit_x = table_pose.block(0, 0, 3, 3) * unit_x;
      table_unit_z = table_pose.block(0, 0, 3, 3) * unit_z;

      if (table_pose.isZero()) 
        continue;
      
      Eigen::Affine3d orient_diff = findTransformBetween2CS(origin, obj_unit_x, obj_unit_z, origin, table_unit_x, table_unit_z);
      Eigen::Matrix3d table_rot= table_pose.block(0, 0, 3, 3);
      concated = orient_diff * concated;

      //printf("OBJECT_HEIGHT: %f\n", OBJECT_HEIGHT);
      //printf("FALSE SCORE: %f\n", FALSE_CHECK_SCORE);

      Eigen::Vector3d to_table =  OBJECT_HEIGHT * 0.5 * table_unit_z;
      concated(0, 3) = tmp_translation(0) + to_table(0);
      concated(1, 3) = tmp_translation(1) + to_table(1);
      //concated(2, 3) = table_pose(2, 3) + OBJECT_HEIGHT / 2;
      concated(2, 3) = table_pose(2, 3) + OBJECT_HEIGHT; // Return the top-surface of object to robot-action

      if ((first_confidence_score < FALSE_CHECK_SCORE))
      {
        float table_width = 0.78; // Y dir
        float table_depth = 0.50; // X dir
        float boundary[] = { ((float)table_pose(0, 3) - table_depth/2), // front_x
                             ((float)table_pose(0, 3) + table_depth/2), // back_x
                             ((float)table_pose(1, 3) - table_width/2), // left_y
                             ((float)table_pose(1, 3) + table_width/2)}; // right_y

        std::cout << "Boundary: " << boundary[0] << " " << boundary[1] << " " << boundary[2] << " "<< boundary[3] << std::endl;
        if (concated(0, 3) < boundary[0] || concated(0, 3) > boundary[1] || concated(1, 3) < boundary[2] || concated(1, 3) > boundary[3])
          continue;

        cv::Point3f mass_center((float)table_pose(0, 3), (float)table_pose(1, 3), (float)table_pose(2, 3));
        /* while (table_arr_listener->positions_vec->size() == 0) {
          ros::Duration(0.1).sleep();
        }
          
        float dist = distance(table_arr_listener->positions_vec->at(0), mass_center);
        if (dist > 0.1) continue; */

        std::cout << "from camera to object: \n" << trans_ork << std::endl;
        std::cout << "from world to table: \n" << table_pose << std::endl;
        std::cout << "from world to camera: \n" << camera_pose << std::endl;
        std::cout << "from world to object: \n" << concated << std::endl;
        
        Eigen::Matrix3d rot = concated.block(0, 0, 3, 3);
        Eigen::Quaterniond pose_q(rot);
        std::cout << "Object quar: "  << pose_q.x() << ", " 
                                      << pose_q.y() << ", " 
                                      << pose_q.z() << ", "
                                      << pose_q.w() << std::endl;
        

        //pose_broadcaster->sendPose(concated, "linemod_pose");
        transform_publisher->fill(concated, match.class_id);
        transform_publisher->publish();
        ROS_INFO("LineMOD Response Published.");

#ifndef DEBUG
        //table_arr_listener->positions_vec->clear();
        req_listener->m_isRequested = false;
        cv::destroyWindow("LINEMOD");
        continue;
#endif
        /*
        // To poseCalculator
        Eigen::Matrix4d cam_pose = Eigen::Matrix4d::Identity();
        
        calculator->grabCurrentPosition(cam_pose);
        calculator->posePushBack(trans_ork);
        //calculator->printPoseArray(calculator->pose_count);
        trans_meaned = calculator->refine();
        */
      }
      
      cv::Matx33d R_concated;
      cv::Vec3d T_concated;
      parseMatxResult(concated, R_concated, T_concated);
      
      //keep the object match
      objs_.push_back(object_recognition_core::db::ObjData(pts_real_ref_temp, pts_real_model_temp, match.class_id, match.similarity, icp_dist, px_ratio_match_inliers, R_concated, T_concated));
      //objs_.push_back(object_recognition_core::db::ObjData(pts_real_ref_temp, point_model_out, match.class_id, match.similarity, icp_dist, px_ratio_match_inliers, R_publish, T_crop));
      ++iter;
    }

    //local non-maxima supression to find the best match at each position
    int count_pass = 0;
    std::vector <object_recognition_core::db::ObjData>::iterator it_o = objs_.begin();
    for (; it_o != objs_.end(); ++it_o)
      if (!it_o->check_done)
      {
#ifndef DEBUG
        if (req_listener->m_isRequested == false) continue;
#endif
        //initialize the object to publish
        object_recognition_core::db::ObjData *o_match = &(*it_o);
        int size_th = static_cast<int>((float)o_match->pts_model.size()*0.85);
        //find the best object match among near objects
        std::vector <object_recognition_core::db::ObjData>::iterator it_o2 = it_o;
        ++it_o2;
        for (; it_o2 != objs_.end(); ++it_o2)
          if (!it_o2->check_done)
            if (cv::norm(o_match->t, it_o2->t) < *th_obj_dist_)
            {
              it_o2->check_done = true;
              if ((it_o2->pts_model.size() > size_th) && (it_o2->icp_dist < o_match->icp_dist))
                o_match = &(*it_o2);
            }

        //perform the final precise icp
        float icp_px_match = 0.0f;
        //float icp_dist = icpCloudToCloud(o_match->pts_ref, o_match->pts_model, o_match->r, o_match->t, icp_px_match, 0);
        float icp_dist = 0.007f;
        if (*verbose_)
          std::cout << o_match->match_class << " " << o_match->match_sim << " icp " << icp_dist << ", ";

        //icp_dist in the same units as the sensor data
        //this distance is used to compute the ratio of inliers (points laying within this distance between the point clouds)
        
        float px_inliers_ratio = 0.;
        if (*verbose_)
          std::cout << " ratio " << o_match->icp_px_match << " or " << px_inliers_ratio << std::endl;
          
        //add points to the clouds
#if LINEMOD_VIZ_PCD
        //pci_real_icpin_model->fill(o_match->pts_model, cv::Vec3b(0,255,0));
        pci_real_icpin_ref->fill(o_match->pts_ref, cv::Vec3b(0,0,255));
        
#endif

        //return the outcome object pose
        pose_result.set_object_id(db_, o_match->match_class);
        pose_result.set_confidence(o_match->match_sim);
        pose_result.set_R(cv::Mat(o_match->r));

        objID_publisher->publish(o_match->match_class);

        o_match->t[0] += 0.0; //-0.025
        o_match->t[1] += 0.0; //-0.005
        o_match->t[2] += 0.0; //0.05
        pose_result.set_T(cv::Mat(o_match->t));
        pose_results_->push_back(pose_result);

        ++count_pass;
      }
    if (*verbose_ && (matches.size()>0))
      std::cout << "matches  " << objs_.size() << " / " << count_pass << " / " << matches.size() << std::endl;

    //publish the point clouds
#if LINEMOD_VIZ_PCD
    //pci_real_icpin_model->publish();
    pci_real_icpin_ref->publish();
#endif
#if LINEMOD_VIZ_IMG
    if (*visualize_) {
      cv::namedWindow("LINEMOD");
      cv::imshow("LINEMOD", display);
      cv::waitKey(1);
    }
#endif
    return ecto::OK;
  }

    /** LINE-MOD detector */
    cv::Ptr<cv::linemod::Detector> detector_;
    // Parameters
    spore<float> threshold_;
    // Inputs
    spore<cv::Mat> color_, depth_;
    /** The calibration matrix of the camera */
    spore<cv::Mat> K_depth_;
    /** The buffer with detected objects and their info */
    std::vector <object_recognition_core::db::ObjData> objs_;
    /** The pointcloud object for registration */
    //spore<PointCloudT::Ptr> pc_model_;

    /** False-positive score of ICP */
    ecto::spore<float> false_positive_threshold_;
    /** The point_cloud path for 2nd ICP. */
    ecto::spore<std::string> json_path_;
    /** True or False to output debug image */
    ecto::spore<bool> visualize_;
    /** True or False to use input rgb image */
    ecto::spore<bool> use_rgb_;
    /** True or False to use input depth image */
    ecto::spore<bool> use_depth_;
    /** Threshold on minimal distance between detected objects */
    ecto::spore<float> th_obj_dist_;
    /** True or False to output debug log */
    ecto::spore<bool> verbose_;
    /** The depth camera frame id*/
    ecto::spore<std::string> depth_frame_id_;
    /** The minimal accepted icp distance*/
    ecto::spore<float> icp_dist_min_;
    /** The minimal percetage of pixels with matching depth*/
    ecto::spore<float> px_match_min_;
    /** The object recognition results */
    ecto::spore<std::vector<PoseResult> > pose_results_;
    /** The rotations, per object and per template */
    std::map<std::string, std::vector<cv::Mat> > Rs_;
    /** The translations, per object and per template */
    std::map<std::string, std::vector<cv::Mat> > Ts_;
    /** The objects distances, per object and per template */
    std::map<std::string, std::vector<float> > distances_;
    /** The calibration matrices, per object and per template */
    std::map<std::string, std::vector<cv::Mat> > Ks_;
    /** The renderer initialized with objects meshes, per object*/
    std::map<std::string, RendererIterator*> renderer_iterators_;
    /** Renderer parameter: the number of points on the sphere */
    int renderer_n_points_;
    /** Renderer parameter: the angle step sampling in degrees*/
    int renderer_angle_step_;
    /** Renderer parameter: the minimum scale sampling*/
    double renderer_radius_min_;
    /** Renderer parameter: the maximum scale sampling*/
    double renderer_radius_max_;
    /** Renderer parameter: the step scale sampling*/
    double renderer_radius_step_;
    /** Renderer parameter: image width */
    int renderer_width_;
    /** Renderer parameter: image height */
    int renderer_height_;
    /** Renderer parameter: near distance */
    double renderer_near_;
    /** Renderer parameter: far distance */
    double renderer_far_;
    /** Renderer parameter: focal length x */
    double renderer_focal_length_x_;
    /** Renderer parameter: focal length y */
    double renderer_focal_length_y_;
  };

} // namespace ecto_linemod

ECTO_CELL(ecto_linemod, ecto_linemod::Detector, "Detector", "Use LINE-MOD for object detection.")

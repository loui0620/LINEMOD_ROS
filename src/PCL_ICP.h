#pragma once

#ifdef __cplusplus
extern "C"
#endif

#ifndef PCL_ICP_H
#define PCL_ICP_H

#include <iostream>
#include <string>
#include <cmath>
#include <complex>
#include <utility> 
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/flann.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/time.h>   // TicToc
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

#include "/usr/local/include/pcl-1.9/pcl/recognition/hv/hv_go.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/gpu/features/features.hpp>

#include <jsoncpp/json/json.h>

// g2o dependencies

#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>

#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/MultiDOFJointState.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Geometry> 


// Custom Message Header
//#include "LinemodResponse.h"
#include <makino_commander/LinemodResponse.h>
#include <makino_commander/PositionArray.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
template <class T>
using Cloud = pcl::PointCloud<T>;

void pcl_debugger(Eigen::Matrix4d &mtx, int template_id);

void pcl_print4x4Matrix(const Eigen::Matrix4d & matrix);

void pcl_computeCentroid(std::vector<cv::Vec3f> &data_in, cv::Vec3f &centroid);

void pcl_parseMatxResult(const Eigen::Matrix4d &matrix, cv::Matx33f &R_ret, cv::Vec3f &T_ret);

void pcl_bundleMatxResult(Eigen::Matrix4d &matrix, const cv::Matx33d &R_ret, const cv::Vec3d &T_ret);

//void pcl_Vector3fToPC(std::vector<cv::Vec3f> &data_in, PointCloudT::Ptr cloud_out, int downsample_step, cv::Vec3d &T_init, bool is_final);
void pcl_Vector3fToPC(std::vector<cv::Vec3f> &data_in, PointCloudT::Ptr cloud_out, int downsample_step);

void pcl_FilterByDistance(PointCloudT::Ptr cloud_in, PointCloudT::Ptr cloud_out, float distance_threshold);

void pcl_PCToVector3f(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<cv::Vec3f> &data_out);

float pcl_runICP(PointCloudT::Ptr cloud_in, PointCloudT::Ptr cloud_target, PointCloudT::Ptr cloud_out, 
        cv::Matx33d &R_ret, cv::Vec3d &T_ret, Eigen::Matrix4d &trans_full, int iterations, bool is_reverse);

float pcl_runICP(PointCloudT::Ptr cloud_in, PointCloudT::Ptr cloud_target, PointCloudT::Ptr cloud_out, 
        pcl::gpu::Octree &scene_gpu,
        cv::Matx33d &R_ret, cv::Vec3d &T_ret, Eigen::Matrix4d &trans_full, int iterations, bool is_reverse);

double checkAlignment(PointCloudT::Ptr scene, pcl::gpu::Octree &scene_gpu, std::vector<PointT> queries);

class poseCalculator{
public:
  poseCalculator();
  void transformCoordinate(Eigen::Matrix4d &object_pose, Eigen::Matrix4d &cam_pose);
  void grabCurrentPosition(Eigen::Matrix4d &cam_pose);
  void posePushBack(Eigen::Matrix4d &mtx);
  void printPoseArray(int &cnt);
  Eigen::Matrix4d refine();

  std::vector<Eigen::Matrix4d> cam_pose_array;
  std::vector<Eigen::Matrix4d> pose_array;
  std::vector<Eigen::Quaterniond> m_quaternion_array;
  
  Eigen::Vector3d m_transVector = Eigen::Vector3d(0, 0, 0);
  int pose_count;
};

class frameListener
{
public:
  frameListener(ros::NodeHandle& nh, const std::string &color_topic, const std::string &depth_topic, const std::string &frame_id);
  void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
  void depthCallback(const sensor_msgs::ImageConstPtr& msg);

  ros::Subscriber rgb_sub;
  ros::Subscriber depth_sub;

  cv::Mat color_from_listener;
  cv::Mat depth_from_listener;
};

class sensorListener
{
public:
  sensorListener(ros::NodeHandle& nh, const std::string &sensor_topic, const std::string &frame_id);
  void sensorCallback(const std_msgs::Float32& msg);

  ros::Subscriber sensor_sub;

  float cachedValue;
  cv::Mat sensor2Camera; // Extrinsic of sensor and camera.
};

class PositionArrListener
{
public:
  PositionArrListener(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id);
  void positionsCallback(const makino_commander::PositionArrayConstPtr &msg);

  ros::Subscriber pose_arr_sub;
  std::vector<cv::Point3f> *positions_vec;
  //std::vector<geometry_msgs::Point> *positions_vec;
};

class requestListener
{
public:
  requestListener(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id);
  void requestCallback(const std_msgs::Empty &msg);
  ros::Subscriber request_sub;

  bool m_isRequested = false;
};

class framePublisher
{
public:
  framePublisher(ros::NodeHandle& nh, const std::string &img_topic, const std::string &frame_id);
  void fillImageAndPublish(cv::Mat &data);

private:
  image_transport::Publisher img_pub;
  sensor_msgs::ImagePtr img_msg;
};

class objectIDPublisher
{
public:
  objectIDPublisher(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id);
  void publish(std::string &str_msg);

  ros::Publisher _obj_id_pub;
};

class tfBroadcaster
{
public:
  void sendPose(Eigen::Matrix4d &pose_mtx, std::string frameName);

  ros::Subscriber tf_pose_sub;
};

class tfListener
{
public:
  void getRelativePose(Eigen::Matrix4d &mtx_out, std::string frame_name);

  tf::TransformListener _listener;
  tf::StampedTransform _pose;
};

class transformPublisher
{
public:
  /**
   * @brief Initializes the point cloud visualization class
   * @param nh The node handle to publish
   * @param topic The topic name to publish
   * @param frame_id The desired frame id */
  transformPublisher(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id);

  // Fill value into publishing-object.
  void fill(Eigen::Matrix4d &transform, std::string object_id);

  //! Cleans the point cloud */
  void clear();

  //! Publishes the point cloud */
  void publish();

  makino_commander::LinemodResponse msg_pack;
  geometry_msgs::TransformStamped mtx_msg;
  Eigen::Matrix4d full_transform;

private:
  ros::Publisher trans_pub_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class extICP
{
public:
	extICP();
	//~extICP(); //destructor
	void icp_debugger();
  void loadModel(std::string &filePath, std::string &object_id);
  PointCloudT::Ptr loadSingleModel(std::string &filePath);

  cv::Rect previous_rect;
	cv::Mat stored_color;
  PointCloudT m_mesh;
  std::map<std::string, PointCloudT> m_mesh_dict;
  std::vector<std::pair<std::string, PointCloudT>> m_mesh_vector;

	Eigen::Matrix4d RT;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#endif // PCL_ICP_H

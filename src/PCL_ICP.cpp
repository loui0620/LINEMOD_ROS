#include "PCL_ICP.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void pcl_debugger(Eigen::Matrix4d &mtx, int template_id)
{
	std::string meshName = "/home/simon/forDebug/ref/mesh_export_" + std::to_string(template_id) + ".ply";
	PointCloudT::Ptr cloud_debug(new PointCloudT);
	pcl::PLYReader reader;
	reader.read("/home/simon/makino_ws/src/linemod/data/JIG_CENTER.ply", *cloud_debug);
	pcl::transformPointCloud(*cloud_debug, *cloud_debug, mtx);
	pcl::io::savePLYFile(meshName, *cloud_debug);
	std::cout << "Mesh file saved." << std::endl;
}

void pcl_computeCentroid(std::vector<cv::Vec3f> &data_in, cv::Vec3f &centroid)
{
	int num = data_in.size();
	cv::Vec3f vec_tmp = {0., 0., 0.};
	for (int i = 0; i < num; ++i)
	{
		cv::Vec3f pts;
		pts = data_in[i];
		vec_tmp(0) += pts(0);
		vec_tmp(1) += pts(1);
		vec_tmp(2) += pts(2);
	}
	centroid = (vec_tmp / num);

}

void pcl_print4x4Matrix(const Eigen::Matrix4d & matrix) {
	printf("----------FROM EXT-ICP INSTANCE--------------\n");
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

// double checkAlignement(PointCloudT::Ptr scene, std::vector<PointT> queries) 
double checkAlignment(PointCloudT::Ptr scene, pcl::gpu::Octree &scene_gpu, std::vector<PointT> queries) 
{
	pcl::gpu::Octree::Queries queries_device;
	queries_device.upload(queries);
	pcl::gpu::NeighborIndices neighbors_gpu;
	// TODO: share cloud GPU
	
	std::cout << "Octree Built." << std::endl;

	scene_gpu.radiusSearch(queries_device, 0.005f, 1, neighbors_gpu);
	
	//scene.octree_gpu.radiusSearch(queries_device, 0.005f, 1, neighbors_gpu);
	queries_device.release();
	std::vector<int> indices, sizes;

	neighbors_gpu.data.download(indices);
	neighbors_gpu.sizes.download(sizes);
	
	double total_diff = 0;
	for (int i = 0; i < queries.size(); i++) {
		if (sizes[i] > 0) {
			auto p = queries[i];
			auto q = scene->at(indices[i]);
			//auto q = scene->points.at(i);
			total_diff += sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z));
			
		} 
		else {
			total_diff += 1;
		}
	}
	return total_diff;
	
}

void pcl_bundleMatxResult(Eigen::Matrix4d &matrix, const cv::Matx33d &R_ret, const cv::Vec3d &T_ret)
{
	for(int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			matrix(i, j) = 0.;
			matrix(i, j) = (double)R_ret(i, j);
		}
	}
	matrix(0, 3) = (double)T_ret(0);
	matrix(1, 3) = (double)T_ret(1);
	matrix(2, 3) = (double)T_ret(2);
	//std::cout << "R_publish" << R_ret << std::endl;
	//std::cout << "T_publish" << T_ret << std::endl;
}

void pcl_parseMatxResult(const Eigen::Matrix4d &matrix, cv::Matx33d &R_ret, cv::Vec3d &T_ret)
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

void pcl_FilterByDistance(PointCloudT::Ptr cloud_in, 
						PointCloudT::Ptr cloud_out, 
						float distance_threshold)
{	
	PointCloudT::Ptr temp_cloud(new PointCloudT);
	// Set distance threshold, if any far noise don't use it
	int num = cloud_in->points.size();
	for (int i = 0; i < num; i++)
	{
		pcl::PointXYZ point;
		if (cloud_in->points.at(i).z < distance_threshold)
		{
			point.x = cloud_in->points.at(i).x;
			point.y = cloud_in->points.at(i).y;
			point.z = cloud_in->points.at(i).z;
			temp_cloud->push_back(point);
		}
	}
	cloud_out = temp_cloud;
}

void pcl_Vector3fToPC(std::vector<cv::Vec3f> &data_in, 
					PointCloudT::Ptr cloud_out, 
					int downsample_step)
{	
	cloud_out->clear();
	int num = data_in.size();
	for (int i = 0; i < num; i++)
	{
		if (i % downsample_step != 0)
		continue;
		pcl::PointXYZ point;
		cv::Vec3f vec;
		vec = data_in.at(i);
		
		point.x = vec(0);
		point.y = vec(1);
		point.z = vec(2);
		cloud_out->push_back(point);
	}
}

/* 

void pcl_Vector3fToPC(std::vector<cv::Vec3f> &data_in, 
					PointCloudT::Ptr cloud_out, 
					int downsample_step, 
					cv::Vec3d &T_init, 
					bool is_final)
{	
	// Set distance threshold, if any far noise don't use it
	float z_far;
	z_far = 3.0;
	
	if (is_final == true)
		z_far = T_init(2) + 0.06;
	
	cloud_out->clear();
	int num = data_in.size();
	for (int i = 0; i < num; i++)
	{
		if (i % downsample_step != 0)
		continue;
		pcl::PointXYZ point;
		cv::Vec3f vec;
		vec = data_in.at(i);
		if (vec(2) < z_far)
		{
			point.x = vec(0);
			point.y = vec(1);
			point.z = vec(2);
			cloud_out->push_back(point);
		}
	}
}
*/

void pcl_PCToVector3f(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<cv::Vec3f> &data_out)
{
    data_out.clear();
    int num = cloud_in->points.size();
    for (int i = 0; i < num; i++)
    {
        cv::Vec3f vec;
        vec(0) = cloud_in->points.at(i).x;
        vec(1) = cloud_in->points.at(i).y;
        vec(2) = cloud_in->points.at(i).z;
        data_out.push_back(vec);
    }
    if (!data_out.empty())
    {
        //std::cout << "PointCloud to Vector3f converted." << std::endl;
        cv::transpose(data_out, data_out);
    }
}

float pcl_runICP(PointCloudT::Ptr cloud_in, PointCloudT::Ptr cloud_target, PointCloudT::Ptr cloud_out, 
				cv::Matx33d &R_ret, cv::Vec3d &T_ret, Eigen::Matrix4d &trans_full, int iterations, bool is_reverse)
{
	pcl::gpu::Octree::PointCloud cloud_gpu;
	pcl::gpu::NeighborIndices neighbors_gpu;
	pcl::gpu::Octree octree_gpu; // remember move out
	// TODO: share cloud GPU
	cloud_gpu.upload(cloud_target->points);
	octree_gpu.setCloud(cloud_gpu);
	octree_gpu.build();
	return pcl_runICP(cloud_in, cloud_target, cloud_out, octree_gpu, R_ret, T_ret, trans_full, iterations, is_reverse);
}
float pcl_runICP(PointCloudT::Ptr cloud_in, PointCloudT::Ptr cloud_target, PointCloudT::Ptr cloud_out, 
				pcl::gpu::Octree &scene_gpu,
				cv::Matx33d &R_ret, cv::Vec3d &T_ret, Eigen::Matrix4d &trans_full, int iterations, bool is_reverse)
{
	float confidence_score = 65535.;
	pcl::console::TicToc time;
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
	
	cv::Matx33d R_tmp;
	cv::Vec3d T_tmp;
	
	time.tic();
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setMaximumIterations(iterations);
	icp.setTransformationEpsilon(1e-8);
	icp.setMaxCorrespondenceDistance(0.12);
	icp.setEuclideanFitnessEpsilon(1);

	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_target);
	icp.align(*cloud_out);
	
	icp.setMaximumIterations(1);  // We set this variable to 1 for the next time we will call .align () function  
	std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << std::endl;

	if (icp.hasConverged())
	{
		std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
		std::cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << std::endl;
		transformation_matrix *= icp.getFinalTransformation().cast<double>();
		//pcl::transformPointCloud(*cloud_in, *cloud_in, transformation_matrix);
		
		std::vector<PointT> pts;
		for (auto &p: cloud_out->points) {
          pts.push_back(PointT(p.x, p.y, p.z));
        }
		time.tic();
		
		confidence_score = checkAlignment(cloud_target, scene_gpu, pts);
		std::cout << "Checker run " << time.toc() << " ms" << std::endl;
		//confidence_score = 1500;
		std::cout << "----- False-Positive Score -----> " << confidence_score << std::endl;
		pcl_parseMatxResult(transformation_matrix, R_tmp, T_tmp);
		
		if (is_reverse == true)
		{
			T_tmp *= -1;
			R_tmp = R_tmp.inv();
		}

		//update the translation matrix: turn to opposite direction at first and then do translation
		T_ret = R_tmp * T_ret;
		
		//do translation
		Eigen::Matrix4d ret_mtx;
		cv::add(T_ret, T_tmp, T_ret);
		//update the rotation matrix
		R_ret = R_tmp * R_ret;
		
		pcl_bundleMatxResult(trans_full, R_ret, T_ret);
		//pcl_print4x4Matrix(trans_full);
	}
	else
	{
		PCL_ERROR("\nICP has not converged.\n");
		//system("pause");
	}
	return confidence_score;
}

sensorListener::sensorListener(ros::NodeHandle& nh, const std::string &sensor_topic, const std::string &frame_id)
{
	sensor_sub = nh.subscribe(sensor_topic, 1, &sensorListener::sensorCallback, this);
	ROS_INFO("Start grabbing distance from Laser-sensor");
}

requestListener::requestListener(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id)
{
	request_sub = nh.subscribe(topic, 1, &requestListener::requestCallback, this);
}

void requestListener::requestCallback(const std_msgs::Empty &msg)
{
	m_isRequested = true;
	//ROS_INFO("Requested, run LineMOD.");
}
void sensorListener::sensorCallback(const std_msgs::Float32& msg)
{
	cachedValue = msg.data;
}

frameListener::frameListener(ros::NodeHandle& nh, const std::string &color_topic, const std::string &depth_topic, const std::string &frame_id)
{
	rgb_sub = nh.subscribe(color_topic, 1, &frameListener::rgbCallback, this);
	depth_sub = nh.subscribe(depth_topic, 1, &frameListener::depthCallback, this);

	ROS_INFO("External subscribe to the OpenNI2 color & depth image topic.");
}

void frameListener::depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
	depth_from_listener.release();
	cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& ex)
    {
        ROS_ERROR("cv_bridge exception in depthcallback: %s", ex.what());
        exit(-1);
    }
    depth_from_listener = cv_ptr->image.clone();
}

void frameListener::rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
	color_from_listener.release();
	cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
		cv_ptr=cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        //cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); // Caution the type here.
    }
    catch (cv_bridge::Exception& ex)
    {
        ROS_ERROR("cv_bridge exception in rgbcallback: %s", ex.what());
        exit(-1);
    }
	
    color_from_listener = cv_ptr->image.clone();
	
}

framePublisher::framePublisher(ros::NodeHandle& nh, const std::string &img_topic, const std::string &frame_id)
{
	image_transport::ImageTransport it(nh);
	img_pub = it.advertise(img_topic, 1);
}

objectIDPublisher::objectIDPublisher(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id)
{
	_obj_id_pub = nh.advertise<std_msgs::String>(topic, 1);
}

void objectIDPublisher::publish(std::string &str_msg)
{
	std_msgs::String msg;
	msg.data = str_msg;
	//ROS_INFO("PUBLISHED OBJECT ID: %s", msg.data.c_str());

	_obj_id_pub.publish(msg);
}

void framePublisher::fillImageAndPublish(cv::Mat &data)
{
	img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", data).toImageMsg();
	img_pub.publish(img_msg);
}

void tfBroadcaster::sendPose(Eigen::Matrix4d &pose_mtx, std::string frameName)
{
	static tf::TransformBroadcaster br;
	tf::Transform transform;
	Eigen::Matrix3d rot = pose_mtx.block(0, 0, 3, 3);
	Eigen::Quaterniond pose_q(rot); // Convert mtx to quart
	
	transform.setOrigin( tf::Vector3(pose_mtx(0, 3), pose_mtx(1, 3), pose_mtx(2, 3)));
	tf::Quaternion tf_q(pose_q.x(), pose_q.y(), pose_q.z(), pose_q.w());

	transform.setRotation(tf_q);
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_footprint", frameName));
}

void tfListener::getRelativePose(Eigen::Matrix4d &mtx_out, std::string frame_name)
{
	ros::Time now = ros::Time(0);
	//ros::Time now = ros::Time::now();
	try{
		_listener.waitForTransform("/base_footprint", frame_name, now, ros::Duration(1.0));
		_listener.lookupTransform("/base_footprint", frame_name, now, _pose);

		Eigen::Matrix4d pose_matrix;
		pose_matrix.setIdentity();
		
		Eigen::Quaterniond quar(_pose.getRotation().getW(),
								_pose.getRotation().getX(), 
								_pose.getRotation().getY(),
								_pose.getRotation().getZ()
								);

		pose_matrix.block(0, 0, 3, 3) = quar.toRotationMatrix();
		
		pose_matrix(0, 3) = _pose.getOrigin().getX();
		pose_matrix(1, 3) = _pose.getOrigin().getY();
		pose_matrix(2, 3) = _pose.getOrigin().getZ();

		mtx_out = pose_matrix;
	}
	catch (tf::TransformException &ex) {
		ROS_ERROR("%s",ex.what());
		//ros::Duration(1.0).sleep();
	}
	
	
}

transformPublisher::transformPublisher(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id):
  trans_pub_(nh.advertise<makino_commander::LinemodResponse>(topic, 1))
{
	mtx_msg.header.frame_id = frame_id;
	mtx_msg.header.stamp = ::ros::Time::now();
	//trans_pub_ = nh.advertise<std_msgs::Float32MultiArray>("linemod_response", 100);
	//trans_pub_ = nh.advertise<geometry_msgs::TransformStamped>("linemod_response", 100);
}

void transformPublisher::fill(Eigen::Matrix4d &mtx_in, std::string object_id)
{
	Eigen::Matrix3d rot = mtx_in.block(0, 0, 3, 3);
	Eigen::Quaterniond q(rot);

	msg_pack.object_id.data = object_id;

	msg_pack.position.x = mtx_in(0, 3);
	msg_pack.position.y = mtx_in(1, 3);
	msg_pack.position.z = mtx_in(2, 3);

	msg_pack.orientation.x = q.x();
	msg_pack.orientation.y = q.y();
	msg_pack.orientation.z = q.z();
	msg_pack.orientation.w = q.w();

	/*
	mtx_msg.transform.translation.x = mtx_in(0, 3);
	mtx_msg.transform.translation.y = mtx_in(1, 3);
	mtx_msg.transform.translation.z = mtx_in(2, 3);
	*/
}

void transformPublisher::publish()
{
	trans_pub_.publish(msg_pack);
	//std::cout << "pub_array PUBLISHED!!!!" << std::endl;
}

extICP::extICP()
{
	RT = Eigen::Matrix4d::Identity();
	m_mesh_dict.erase(m_mesh_dict.begin(), m_mesh_dict.end());
}

void extICP::loadModel(std::string &filePath, std::string &object_ID)
{
	PointCloudT::Ptr cloud_temp(new PointCloudT);
	pcl::PLYReader reader;
	reader.read(filePath, *cloud_temp);
	//m_mesh_dict[object_ID] = PointCloudT(*cloud_temp);
	
	std::pair<std::string, PointCloudT> label_mesh;
	label_mesh.first = object_ID;
	label_mesh.second = PointCloudT(*cloud_temp);
	m_mesh_vector.push_back(label_mesh);

	std::cout << "Mesh vector size; " << m_mesh_vector.size() << "\n";
}

PointCloudT::Ptr extICP::loadSingleModel(std::string &filePath)
{
	PointCloudT::Ptr cloud_temp(new PointCloudT);
	pcl::PLYReader reader;
	reader.read(filePath, *cloud_temp);
	return cloud_temp;
}

poseCalculator::poseCalculator()
{
	m_transVector = Eigen::Vector3d(0, 0, 0);
	pose_count = 0;
}

void poseCalculator::posePushBack(Eigen::Matrix4d &mtx)
{
	pose_array.push_back(mtx);
	pose_count += 1;
}

void poseCalculator::grabCurrentPosition(Eigen::Matrix4d &mtx)
{
	cam_pose_array.push_back(mtx);
}

void poseCalculator::printPoseArray(int &cnt)
{
	if (cnt == 4)
	{
		for (int i = 0; i < pose_array.size(); ++i)
		{
			pcl_print4x4Matrix(pose_array[i]);
		}
		pose_count = 0;
		pose_array.clear();
	}
}

void separateRT(Eigen::Matrix4d &data_in, Eigen::Matrix3d &R, Eigen::Vector3d &t)
{
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			data_in(i, j) = R(i, j);
		}
	}
	t(0) = data_in(0, 3);
	t(1) = data_in(1, 3);
	t(2) = data_in(2, 3);
}

void bundleRT(Eigen::Matrix4d &data_out, Eigen::Matrix3d &R_in, Eigen::Vector3d &t_in)
{
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			data_out(i, j) = R_in(i, j);
		}
	}
	data_out(0, 3) = t_in(0);
	data_out(1, 3) = t_in(1);
	data_out(2, 3) = t_in(2);
}

void poseCalculator::transformCoordinate(Eigen::Matrix4d &object_pose, Eigen::Matrix4d &cam_pose)
{
	Eigen::Quaterniond Q_obj, Q_cam;
	Eigen::Matrix3d R_obj, R_cam;
	Eigen::Vector3d t_obj, t_cam;

	separateRT(object_pose, R_obj, t_obj);
	separateRT(cam_pose, R_cam, t_cam);

	// Quarternion format
	//Q_obj = R_obj;
	//Q_cam = R_cam;

	t_obj = ( R_cam * t_obj ) + t_cam;
	//update the rotation matrix
	R_obj = R_cam * R_obj;
	
	Q_obj = Eigen::Quaterniond(R_obj);
	m_quaternion_array.push_back(Q_obj);

	std::cerr << "[INFO ] Quaternion: \n" << Q_obj.coeffs() << std::endl;
	m_transVector = m_transVector + t_obj;

	Eigen::Matrix4d returnPose;
	bundleRT(returnPose, R_obj, t_obj);

	object_pose = returnPose;
}

Eigen::Matrix4d poseCalculator::refine()
{
	Eigen::Matrix4d finale  = Eigen::Matrix4d::Identity();
	if (pose_count == 3) // camera pose number
	{
		// Calculating from very last pose to 1-st pose.
		for (int i = 0; i < pose_array.size(); ++i)
		{
			Eigen::Quaterniond pose_quart;
			Eigen::Matrix4d objectPose = pose_array[i];
			
			transformCoordinate(pose_array[i], cam_pose_array[i]);
			pcl_print4x4Matrix(objectPose);
			finale = objectPose;
		}
		// MEAN translation ONLY
		
		m_transVector /= 3;
		finale(0, 3) = m_transVector(0);
		finale(1, 3) = m_transVector(1);
		finale(2, 3) = m_transVector(2);
		
		std::cerr << "FINALE: ";
		pcl_print4x4Matrix(finale);
		// Clear Pose objects.
		pose_count = 0;
		pose_array.clear();
		m_transVector = Eigen::Vector3d(0, 0, 0);
	}
	
	return finale;
}

PositionArrListener::PositionArrListener(ros::NodeHandle& nh, const std::string &topic, const std::string &frame_id)
{
	//positions_vec = new std::vector<geometry_msgs::Point>();
	positions_vec = new std::vector<cv::Point3f>();
	pose_arr_sub = nh.subscribe(topic, 1, &PositionArrListener::positionsCallback, this);
}

void PositionArrListener::positionsCallback(const makino_commander::PositionArrayConstPtr &msg)
{ 
	int i = 0;
	ROS_INFO("positionsCallback entered.\n");
	
	if (msg->position_array.size() != 0)
	{
		printf("has value\n");
		for(int i = 0; i < msg->position_array.size(); i++)
		{
			cv::Point3f point(msg->position_array[i].x, msg->position_array[i].y, msg->position_array[i].z);
			positions_vec->push_back(point);
			//positions_vec->push_back(msg->position_array[i]);
			std::cout << msg->position_array[i].x << std::endl;
			std::cout << msg->position_array[i].y << std::endl;
			std::cout << msg->position_array[i].z << std::endl;
		}
	}
	
}
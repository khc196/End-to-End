#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include <iostream>
#include <fstream>
#include <ctime>
#include <queue>
#include <cv.h>
#include <unistd.h>
#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <string.h>
#include <sys/time.h>
#include "inverseMapping.hpp"
#include "dbscan.h"
using namespace cv;
using namespace std;
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))  
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))  

const CvScalar COLOR_BLUE = CvScalar(255, 0, 0);
const CvScalar COLOR_RED = CvScalar(0, 0, 255);
const CvScalar COLOR_GREEN = CvScalar(0, 255, 0);

const Vec3b RGB_WHITE_LOWER = Vec3b(100, 100, 100);
const Vec3b RGB_WHITE_UPPER = Vec3b(255, 255, 255);
const Vec3b RGB_YELLOW_LOWER = Vec3b(225, 180, 0);
const Vec3b RGB_YELLOW_UPPER = Vec3b(255, 255, 170);
const Vec3b HSV_YELLOW_LOWER = Vec3b(20, 20, 130);
const Vec3b HSV_YELLOW_UPPER = Vec3b(40, 140, 255);

const Vec3b HLS_YELLOW_LOWER = Vec3b(20, 120, 80);
const Vec3b HLS_YELLOW_UPPER = Vec3b(45, 200, 255);

string to_string(int n) {
	stringstream s;
	s << n;
	return s.str();
}
void printResults(vector<cPoint>& points, int num_points)
{
    int i = 0;
    printf("Number of points: %u\n"
        " x     y     z     cluster_id\n"
        "-----------------------------\n"
        , num_points);
    while (i < num_points)
    {
          printf("%5d %5d %5d: %d\n",
                 points[i].x,
                 points[i].y, points[i].z,
                 points[i].clusterID);
          ++i;
    }
}

class Lane_Detector {
protected:

	float left_slope;
	float right_slope;
	float left_length;
	float right_length;

	bool left_error;
	bool right_error;
	Mat input_left, input_right;

	int left_error_count;
	int right_error_count;

	Mat img_hsv, filterImg, binaryImg, initROI,
		mask, cannyImg, houghImg;

    int vanishing_point_x;
    int vanishing_point_y;
    int* ipm_table;
    
	void base_ROI(Mat& img, Mat& img_ROI);
	void v_roi(Mat& img, Mat& img_ROI, const Point& p1, const Point& p2);
	void region_of_interest_L(Mat& img, Mat& img_ROI);
	void region_of_interest_R(Mat& img, Mat& img_ROI);
	bool hough_left(Mat& img, Point* p1, Point* p2);
	bool hough_right(Mat& img, Point* p1, Point* p2);
	float get_slope(const Point& p1, const Point& p2);
	int position(const Point P1, const Point P2);
	int argmax(int* arr, int size);
	vector<double> polyfit(int size, vector<Point> vec_p, int degree);
	vector<int> filter_line(vector<int> vec_nonzero);
public:
	Point p1, p2, p3, p4;
	Mat originImg_left;
	Mat originImg_right;
	Lane_Detector() {}
	void init();
	void operate(Mat originImg);
	float get_left_slope();
	float get_right_slope();
	float get_left_length();
	float get_right_length();
	bool is_left_error();
	bool is_right_error();
	bool get_intersectpoint(const Point& AP1, const Point& AP2,
		const Point& BP1, const Point& BP2, Point* IP);
};

bool Lane_Detector::is_left_error() {
	return left_error;
}
bool Lane_Detector::is_right_error() {
	return right_error;
}
int Lane_Detector::position(const Point P1, const Point P3) {
	float x_L;
	float x_R;
	const float y = 480;

	x_L = (y - P1.y + left_slope * P1.x) / left_slope;
	left_length = 320 - x_L;

	x_R = (y - P3.y + right_slope * P3.x) / right_slope;
	right_length = x_R;
}

float Lane_Detector::get_left_length() {
	return left_length;
}

float Lane_Detector::get_right_length() {
	return right_length;
}

float Lane_Detector::get_left_slope() {
	return left_slope;
}
float Lane_Detector::get_right_slope() {
	return right_slope;
}

void Lane_Detector::init() {
	string path = "/home/foscar/ISCC_Videos/";
	struct tm* datetime;
	time_t t;
	t = time(NULL);
	datetime = localtime(&t);
	string s_t = path.append(to_string(datetime->tm_year + 1900)).append("-").append(to_string(datetime->tm_mon + 1)).append("-").append(to_string(datetime->tm_mday)).append("_").append(to_string(datetime->tm_hour)).append(":").append(to_string(datetime->tm_min)).append(":").append(to_string(datetime->tm_sec)).append(".avi");

	mask = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

	left_error = true;
	right_error = true;
	left_length = 0;
	right_length = 0;
	left_error_count = 0;
	right_error_count = 0;

	vanishing_point_x = 320;
	vanishing_point_y = 235;
	ipm_table = new int[DST_REMAPPED_WIDTH * DST_REMAPPED_HEIGHT];
    build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT, 
                    DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, 
                    vanishing_point_x, vanishing_point_y, ipm_table);
}

void Lane_Detector::operate(Mat originImg) {
	Mat imgray;
    Mat imremapped = Mat(DST_REMAPPED_HEIGHT, DST_REMAPPED_WIDTH, CV_8UC1);
	imremapped = imremapped(Rect(0, 0, 200, 200));
	cvtColor(originImg, imgray, CV_BGR2GRAY);
    inverse_perspective_mapping(DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, imgray.data, ipm_table, imremapped.data);
	imremapped = imremapped(Rect(30, 0, 140, 120));
	Canny(imremapped, cannyImg, 70, 210);
	int kernel_data[] = {2, -1, -1, -1, 2, -1, -1, -1, 2};
	Mat roi_l = cannyImg(Rect(0, 0, 70, 120));
	Mat roi_r = cannyImg(Rect(70, 0, 70, 120));
	Mat dilated_l, dilated_r;
	
	morphologyEx(roi_l, dilated_l, MORPH_CLOSE, Mat(9,9, CV_8U, Scalar(1)));
	morphologyEx(roi_r, dilated_r, MORPH_CLOSE, Mat(9,9, CV_8U, Scalar(1)));
	// imshow("closed_l", dilated_l);
	// imshow("closed_r", dilated_r);
	int histogram_l[4][70] = {0};
	int histogram_r[4][70] = {0};
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 30; j++){
			for(int k = 0; k < 70; k++){
				histogram_l[i][k] += dilated_l.data[(i * 30 + j) * dilated_l.step + k];
				histogram_r[i][k] += dilated_r.data[(i * 30 + j) * dilated_r.step + k];
			}
		}
	}
	int max_arg_l[4], max_arg_r[4];
	
	cvtColor(imremapped, imremapped, COLOR_GRAY2BGRA);
	int zero_count_l = 0, zero_count_r = 0;
	vector<Point> points_l, points_r;
	vector<double> poly_left, poly_right;
	for(int i = 0; i < 4; i++){
		for(int j = 1; j <= 22; j+=2){
			line(imremapped, Point(7*(j), i*30 + 30), Point(7*(j+1), i*30 + 30), COLOR_RED, 1, CV_AA);
		}
		max_arg_l[i] = argmax(histogram_l[i], 70);
		max_arg_r[i] = argmax(histogram_r[i], 70);
		
		if(max_arg_l[i] != -1){
			//circle(imremapped, Point(max_arg_l[i], i * 30 + 15), 3, Scalar(255, 255, 0), -1);
			points_l.push_back(Point(i * 30 + 15, max_arg_l[i]));
		}
		if(max_arg_r[i] != -1){
			//circle(imremapped, Point(max_arg_r[i]+70, i * 30 + 15), 3, Scalar(0, 255, 255), -1);
			points_r.push_back(Point(i * 30 + 15, max_arg_r[i]));
		}
	}	
	Mat forcurve;
	//cvtColor(imremapped, imremapped, COLOR_GRAY2BGRA);
	forcurve = imremapped.clone();
	int degree_l = 1, degree_r = 1;
	if(points_l.size() == 4)
		degree_l = 3;
	if(points_r.size() == 4)
		degree_r = 3;
	poly_left = polyfit(points_l.size(), points_l, degree_l);
	poly_right = polyfit(points_r.size(), points_r, degree_r);
	for(int i = 0; i < 120; i++){
		int x = 0;
		for(int j = 0; j <= degree_l; j++)
			x += poly_left[j] * pow(i, j);
		if(x >= 0)
		circle(forcurve, Point(x, i), 3, Scalar(255, 255, 0), -1);
	}
	for(int i = 0; i < 120; i++){
		int x = 0;
		for(int j = 0; j <= degree_r; j++)
			x += poly_right[j] * pow(i, j);
		if(x >= 0)
		circle(forcurve, Point(x+70, i), 3, Scalar(0, 255, 255), -1);
	}
	addWeighted(forcurve, 0.3, imremapped, 0.7, 0.0, imremapped);
	// vector<int> nonzero_l_x, nonzero_l_y, nonzero_r_x, nonzero_r_y;
	// vector<Point> nonzero_l_p, nonzero_r_p; 
	// for(int i = 0; i < 120; i++) {
	// 	for(int j = 0; j < 70; j++){
	// 		histogram_l[j] += dilated_l.data[i * dilated_l.step + j];
	//  		histogram_r[j] += dilated_r.data[i * dilated_r.step + j];
	// 		if(dilated_l.data[i * dilated_l.step + j] != 0){
	// 			nonzero_l_y.push_back(i);
	// 			nonzero_l_x.push_back(j);
	// 			Point temp = Point(j, i);
	// 			nonzero_l_p.push_back(temp);
	// 		}
	// 		if(dilated_r.data[i * dilated_l.step + j] != 0){
	// 			nonzero_r_y.push_back(i);
	// 			nonzero_r_x.push_back(j);
	// 			Point temp = Point(j, i);
	// 			nonzero_r_p.push_back(temp);
	// 		}
	// 	}
	// }
	// //imshow("dilated_r", dilated_r);
	// int current_left_x = argmax(histogram_l, 70);
	// int current_right_x = argmax(histogram_r, 70);
	// int window_height = 10;
	// int minpixel = 3;
	// Mat forcurve;
	// cvtColor(imremapped, imremapped, COLOR_GRAY2BGRA);
	// forcurve = imremapped.clone();
	// vector<Point> vec_left, vec_right;
	// int noneCount_l = 0, noneCount_r = 0;
	// for(int i = 0; i < 14; i++) {
	// 	int win_y_high = 0 + window_height * (i+1);
	// 	int win_y_low = 0 + window_height * (i);
	// 	int center_y = (win_y_high + win_y_low) / 2;
	// 	int nonzero_l_sum = 0, nonzero_r_sum = 0, nonzero_l_size = 0, nonzero_r_size = 0;
	// 	for(int j = 0; j < nonzero_l_x.size(); j++){
	// 		if(nonzero_l_y[j] < win_y_high && nonzero_l_y[j] >= win_y_low){
	// 			nonzero_l_sum += nonzero_l_x[j];
	// 			nonzero_l_size++;
	// 			nonzero_l_x.erase(nonzero_l_x.begin() + j);
	// 			noneCount_l = 0;
	// 		}
	// 		else{
	// 			noneCount_l++;
	// 		}
	// 	}
	// 	for(int j = 0; j < nonzero_r_x.size(); j++){
	// 		if(nonzero_r_y[j] < win_y_high && nonzero_r_y[j] >= win_y_low){
	// 			nonzero_r_sum += nonzero_r_x[j];
	// 			nonzero_r_size++;
	// 			nonzero_r_x.erase(nonzero_r_x.begin() + j);
	// 		}
	// 	}
	// 	if(nonzero_l_size >= minpixel){
	// 		int nonzero_l_avg = nonzero_l_sum / nonzero_l_size;
	// 		if(nonzero_l_avg < current_left_x+15 && nonzero_l_avg > current_left_x-40) {
	// 			current_left_x = nonzero_l_avg;
	// 			noneCount_l = 0;
	// 		}
	// 		else{
	// 			noneCount_l++;
	// 		}
	// 	}
	// 	if(nonzero_r_size >= minpixel){
	// 		int nonzero_r_avg = nonzero_r_sum / nonzero_r_size;
	// 		if(nonzero_r_avg < current_right_x+40 && nonzero_r_avg > current_right_x-15) {
	// 			current_right_x = nonzero_r_avg;
	// 			noneCount_r = 0;
	// 		}
	// 		else {
	// 			noneCount_r++;
	// 		}
	// 	}
	// 	//rectangle(forcurve, Point(current_left_x-10, win_y_low), Point(current_left_x+10, win_y_high), Scalar(255,0, 0), 2);
	// 	//rectangle(forcurve, Point(current_right_x+60, win_y_low), Point(current_right_x+80, win_y_high), Scalar(255, 0, 0), 2);
	// 	vec_left.push_back(Point(current_left_x, center_y));
	// 	vec_right.push_back(Point(current_right_x+70, center_y));
	// }
	// int degree = 1;
	// Mat curve, curve2;
	// curve = Mat(vec_left, true);
	// curve2 = Mat(vec_right, true);
	// if(nonzero_l_p.size() > 60)
	// 	polylines(forcurve, curve, true, Scalar(255, 255, 0), 15);
	// if(nonzero_r_p.size() > 60)
	// 	polylines(forcurve, curve2, true, Scalar(0, 255, 255), 15);
	// addWeighted(forcurve, 0.3, imremapped, 0.7, 0.0, imremapped);
	// DBSCAN ds(10, 2800.0, nonzero_l_p);
	// DBSCAN ds2(10, 2800.0, nonzero_r_p);
	// DBSCAN ds(4, 1.0, nonzero_l_p);
	// DBSCAN ds2(4, 1.0, nonzero_r_p);
	// ds.run();
	// ds2.run();
	// vector<cPoint> nonzero_l_p_f, nonzero_r_p_f;
	// int countIDs[100] = { 0, };
	// int countIDs2[100] = { 0, };
	// int i = 1;
	// while (i < ds.getTotalPointSize())
    // {
	// 	countIDs[ds.m_points[i].clusterID]++;
	// 	i++;
    // }
	// int j = 1;
	// while (j < ds2.getTotalPointSize())
    // {
    //     countIDs2[ds2.m_points[j].clusterID]++;
	// 	j++;
    // }
	// i = 1;
	// i = 1;
	// while(countIDs[i] > 0){
	// 	countIDs[i] += 3 * avgX[i];
	// 	i++;
	// }
	// i = 1;
	// while(countIDs2[i] > 0){
	// 	countIDs2[i] -= 3 * avgX2[i];
	// 	i++;
	// }
	// i = 1;
	// int maxCount = 0, maxID = 0;
	// while(countIDs[i] > 0){
	// 	if(maxCount < countIDs[i]){
	// 		maxCount = countIDs[i];
	// 		maxID = i;
	// 	}
	// 	i++;
	// }
	// i = 1;
	// int maxCount2 = 0, maxID2 = 0;
	// while(countIDs2[i] > 0){
	// 	if(maxCount2 < countIDs2[i]){
	// 		maxCount2 = countIDs2[i];
	// 		maxID2 = i;
	// 	}
	// 	i++;
	// }
	// i = 0;
	// while(i < ds.getTotalPointSize()){
	// 	// if(ds.m_points[i].clusterID == maxID)
	// 	if(ds.m_points[i].clusterID > 0)
	// 		nonzero_l_p_f.push_back(ds.m_points[i]);
	// 	i++;
	// } 
	// i = 0;
	// while(i < ds2.getTotalPointSize()){
	// 	// if(ds2.m_points[i].clusterID == maxID2)
	// 	if(ds2.m_points[i].clusterID > 0)
	// 		nonzero_r_p_f.push_back(ds2.m_points[i]);
	// 	i++;
	// }
	// // printf("=============left===============\n");
	// //vector<double> poly_left = polyfit(nonzero_l_p_f.size(), nonzero_l_p_f, 1);
	// // printf("============right===============\n");
	// //vector<double> poly_right = polyfit(nonzero_r_p_f.size(), nonzero_r_p_f, 1);
	// Mat left_lane(140, 100, CV_8UC3);
	// Mat right_lane(140, 100, CV_8UC3);
	// left_lane = Scalar(0, 0, 0);
	// right_lane = Scalar(0, 0, 0);
	// vector<Point> points, points2;
	// cvtColor(imremapped, imremapped, COLOR_GRAY2BGRA);
	// Mat forcurve = imremapped.clone();
	// for(int i = 0; i < nonzero_l_p_f.size(); i++){
	// 	//points.push_back(Point(nonzero_l_p_f[i].x, nonzero_l_p_f[i].y));
	// 	//int x = nonzero_l_p_f[i].x;
	// 	//int y = poly_left[0] * pow(x, 0) + poly_left[1] * pow(x, 1);// + poly_left[2] * pow(x, 2) + poly_left[3] * pow(x, 3);
	// 	//points.push_back(Point(x, int(y)));
	// 	//circle(forcurve, Point(x, int(y)), 3, Scalar(255, 255, 0), -1);
	// 	circle(forcurve, Point(nonzero_l_p_f[i].x, nonzero_l_p_f[i].y), 3, Scalar(255, 255, 0), -1);
	// }
	// for(int i = 0; i < nonzero_r_p_f.size(); i++){
	// 	//points2.push_back(Point(nonzero_r_p_f[i].x+100, nonzero_r_p_f[i].y));
	// 	// int x = nonzero_r_p_f[i].x;
	// 	// int y = poly_right[0] * pow(x, 0) + poly_right[1] * pow(x, 1);// + poly_right[2] * pow(x, 2) + poly_right[3] * pow(x, 3);
	// 	// points2.push_back(Point(x+100, int(y)));
	// 	// circle(forcurve, Point(x+100, int(y)), 3, Scalar(0, 255, 255), -1);
	// 	circle(forcurve, Point(nonzero_r_p_f[i].x+100, nonzero_r_p_f[i].y), 3, Scalar(0, 255, 255), -1);
	// }
	// // for(int i = 0; i < 100; i++){
	// // 	double y = 0;
	// // 	for(int j = 0; j <= 3; j++){
	// // 		y += poly_left[j] * pow(i, j);
	// // 	}
	// // }
	// // Mat curve(points, true);
	// // Mat curve2(points2, true);
	// // //polylines(left_lane, curve, true, Scalar(255, 255, 0), 15);
	// // 
	
	
	// // polylines(forcurve, curve, true, Scalar(255, 255, 0), 15);
	// // polylines(forcurve, curve2, true, Scalar(0, 255, 255), 15);

	// addWeighted(forcurve, 0.3, imremapped, 0.7, 0.0, imremapped);
	// int min_num_pixel = 1;
	// for(int i = 0; i < 140; i++){
	// 	for(int j = 0; j < 100; j++) {
	// 		histogram_l[j] += dilated_l.data[i * dilated_l.step + j];
	// 		histogram_r[j] += dilated_r.data[i * dilated_r.step + j];
	// 		if(dilated_l.data[i * dilated_l.step + j] != 0){
	// 			nonzero_l_y.push_back(i);
	// 			nonzero_l_x.push_back(j);
	// 		}
	// 		if(dilated_r.data[i * dilated_l.step + j] != 0){
	// 			nonzero_r_y.push_back(i);
	// 			nonzero_r_x.push_back(j);
	// 		}
	// 	}
	// }
	// int current_left_x = argmax(histogram_l, 100);
	// int current_right_x = argmax(histogram_r, 100);
	// int window_height = 20;
	// for(int i = 0; i < 7; i++) {
	// 	int win_y_low = 140 - window_height * (i+1);
	// 	int win_y_high = 140 - window_height * (i);
	// 	int win_leftx_min = current_left_x - 15;
	// 	int win_leftx_max = current_left_x + 15;
	// 	int win_rightx_min = current_right_x - 15;
	// 	int win_rightx_max = current_right_x + 15;
	// 	if(win_leftx_max > 100){
	// 		win_leftx_min = 70;
	// 		win_leftx_max = 100;
	// 	}
	// 	if(win_rightx_max > 100){
	// 		win_rightx_min = 70;
	// 		win_rightx_max = 100;
	// 	}
	// 	rectangle(dilated_l, Point(win_leftx_min, win_y_low), Point(win_leftx_max, win_y_high), Scalar(255,255,255), 2);
	// 	rectangle(dilated_r, Point(win_rightx_min, win_y_low), Point(win_rightx_max, win_y_high), Scalar(255,255,255), 2);
	// 	vector<int> left_window_x_inds;
	// 	vector<int> left_window_y_inds;
	// 	vector<int> right_window_x_inds;
	// 	vector<int> right_window_y_inds;
	// 	for(int l = 0; l < nonzero_l_x.size(); l++){
	// 		if(nonzero_l_x.at(l) > win_leftx_min && nonzero_l_x.at(l) < win_leftx_max 
	// 		&& nonzero_l_y.at(l) > win_y_low && nonzero_l_y.at(l) < win_y_high){
	// 			left_window_x_inds.push_back(nonzero_l_x.at(l));
	// 			left_window_y_inds.push_back(nonzero_l_y.at(l));
	// 		}
	// 	}
	// 	for(int l = 0; l < nonzero_r_x.size(); l++){
	// 		if(nonzero_r_x.at(l) > win_rightx_min && nonzero_r_x.at(l) < win_rightx_max 
	// 		&& nonzero_r_y.at(l) > win_y_low && nonzero_r_y.at(l) < win_y_high){
	// 			right_window_x_inds.push_back(nonzero_r_x.at(l));
	// 			right_window_y_inds.push_back(nonzero_r_y.at(l));
	// 		}
	// 	}
	// 	printf("%d : %d\n", i, left_window_x_inds.size());
	// 	if(left_window_x_inds.size() >= min_num_pixel){
	// 		current_left_x = 0;
	// 		for(int l = 0; l < nonzero_l_x.size(); l++){
	// 			current_left_x += nonzero_l_x.at(l);
	// 		}
	// 		current_left_x /= nonzero_l_x.size();
	// 	}
	// 	if(right_window_x_inds.size() >= min_num_pixel){
	// 		current_right_x = 0;
	// 		for(int l = 0; l < nonzero_r_x.size(); l++){
	// 			current_right_x += nonzero_r_x.at(l);
	// 		}
	// 		current_right_x /= nonzero_r_x.size();
	// 	}
 	// }
	
	//hconcat(dilated_l, dilated_r, result);
	
	//imshow("left result", left_lane);
	//imshow("left", dilated_l);
	// imshow("Result", result);
	// int i = 200;
	// int j = 180;
	// while (j >= 0){
	// 	int histogram[200];
	// 	for(int i = 0; i < 200; i++) {
	// 		for(int j = 0; j < 140; j++) {
	// 			histogram[i] += cannyImg.at(i, j);
	// 		}
	// 	}
	// 	int left_peak = argmax(histogram, 100);
	// 	int right_peak = argmax(histogram + 4 * 100, 100)
	// 	i -= 20;
	// 	j -= 20;
	// }
	
	// int midpoint = 100;
	// int startleftX, startrightX, current_leftX, current_rightX;
	// int num_windows = 7;
	// int window_height = 20;
	// startleftX = argmax(histogram, 100);
	// startrightX = argmax(histogram + 4 * 100, 100);
	// current_leftX = startleftX;
	// current_rightX = startrightX;

	// hough_left(roi_l, &p1, &p2);
	// hough_right(roi_r, &p3, &p4);
	// p3.x += 100;
	// p4.x += 100;
	// line(imremapped, p1, p2, COLOR_RED, 4, CV_AA);
	// line(imremapped, p3, p4, COLOR_RED, 4, CV_AA);
	resize(originImg, originImg, Size(160, 120), 0, 0, CV_INTER_NN);
	cvtColor(originImg, originImg, CV_BGR2BGRA);
	Mat result, result2;
	hconcat(cannyImg, dilated_l, dilated_l);
	hconcat(dilated_l, dilated_r, result);
	cvtColor(result, result, CV_GRAY2BGRA);
	hconcat(result, imremapped, result2);
	hconcat(originImg, result2, result2);
	//imshow("Img", imremapped);
	imshow("Img", result2);
	waitKey(0);
}
vector<int> Lane_Detector::filter_line(vector<int> vec_nonzero){

}
vector<double> Lane_Detector::polyfit(int size, vector<Point> vec_p, int degree){
	int n = degree;
	double X[2*n+1];
	printf("%d\n", size);
	for(int i = 0; i<2*n+1; i++){
		X[i]=0;
		for(int j = 0; j<size; j++){
			X[i]=X[i]+pow(vec_p[j].x, i);
		}
	}
	double B[n+1][n+2], a[n+1];
	for(int i = 0; i<=n; i++){
		for(int j = 0; j<=n; j++){
			B[i][j]=X[i+j];
		}
	}
	double Y[n+1];
	for(int i = 0; i<n+1; i++){
		Y[i] = 0;
		for(int j = 0; j < size; j++){
			Y[i]=Y[i]+pow(vec_p[j].x, i)*vec_p[j].y;
		}
	}
	for (int i = 0; i<=n; i++){
        B[i][n+1]=Y[i];
	}                
    n=n+1;    
	for(int i = 0; i<n; i++){
		for(int k = i+1; k < n; k++){
			if(B[i][i] < B[k][i]){
				for(int j=0; j<=n; j++){
					double temp = B[i][j];
					B[i][j] = B[k][j];
					B[k][j] = temp;
				}
			}
		}
	}
	for(int i = 0; i<n-1; i++){
		for(int k = i+1; k<n; k++){
			double t = B[k][i]/B[i][i];
			for (int j = 0; j <= n; j++){
				B[k][j] = B[k][j] - t*B[i][j];
			}
		}
	} 
	for(int i = n-1; i >= 0; i--){
		a[i] = B[i][n];
		for(int j = 0; j<n; j++){
			if(j != i){
				a[i] = a[i] - B[i][j] * a[j];
			}
		}
		a[i] = a[i] / B[i][i];
	}
	for(int i = n-1; i >= 0; i--){
		printf("%fx^%d", a[i], i);
		if(i != 0){
			printf(" + ");
		}
		else{
			printf("\n");
		}
	}
	vector<double> result;
	for(int i = 0; i < n; i++){
		result.push_back(a[i]);
	}
	return result;
}	
float Lane_Detector::get_slope(const Point& p1, const Point& p2) {

	float slope;

	if (p2.y - p1.y != 0.0) {
		slope = ((float)p2.y - (float)p1.y) / ((float)p2.x - (float)p1.x);
	}
	return slope;
}

bool Lane_Detector::get_intersectpoint(const Point& AP1, const Point& AP2,
	const Point& BP1, const Point& BP2, Point* IP)
{
	double t;
	double s;
	double under = (BP2.y - BP1.y)*(AP2.x - AP1.x) - (BP2.x - BP1.x)*(AP2.y - AP1.y);
	if (under == 0) return false;

	double _t = (BP2.x - BP1.x)*(AP1.y - BP1.y) - (BP2.y - BP1.y)*(AP1.x - BP1.x);
	double _s = (AP2.x - AP1.x)*(AP1.y - BP1.y) - (AP2.y - AP1.y)*(AP1.x - BP1.x);

	t = _t / under;
	s = _s / under;

	if (t<0.0 || t>1.0 || s<0.0 || s>1.0) return false;
	if (_t == 0 && _s == 0) return false;

	IP->x = AP1.x + t * (double)(AP2.x - AP1.x);
	IP->y = AP1.y + t * (double)(AP2.y - AP1.y);

	return true;
}

bool Lane_Detector::hough_left(Mat& img, Point* p1, Point* p2) {

	vector<Vec2f> linesL;

	int count = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
	int threshold = 40;

	for (int i = 10; i > 0; i--) {
		HoughLines(img, linesL, 1, CV_PI / 180, threshold, 0, 0, 0, CV_PI);
		int clusterCount = 2;
		Mat h_points = Mat(linesL.size(), 1, CV_32FC2);
		Mat labels, centers;
		
		if (linesL.size() > 1) {
			for (size_t i = 0; i < linesL.size(); i++) {
				count++;
				float rho = linesL[i][0];
				float theta = linesL[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				// cout << "x0, y0 : " << rho << ' ' << theta << endl;
				h_points.at<Point2f>(i, 0) = Point2f(rho, (float)(theta * 100));
			}
			
			// kmeans(h_points, clusterCount, labels,
			// 	TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
			// 	3, KMEANS_RANDOM_CENTERS, centers);

			Point mypt1 = centers.at<Point2f>(0, 0);

			float rho = mypt1.x;
			float theta = (float)mypt1.y / 100;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;

			// cout << "pt : " << mypt1.x << ' ' << mypt1.y << endl;

			int _x1 = int(x0 + 1000 * (-b));
			int _y1 = int(y0 + 1000 * (a));
			int _x2 = int(x0 - 1000 * (-b));
			int _y2 = int(y0 - 1000 * (a));

			x1 += _x1;
			y1 += _y1;

			x2 += _x2;
			y2 += _y2;

			Point mypt2 = centers.at<Point2f>(1, 0);

			rho = mypt2.x;
			theta = (float)mypt2.y / 100;
			a = cos(theta), b = sin(theta);
			x0 = a * rho, y0 = b * rho;

			// cout << "pt : " << mypt2.x << ' ' << mypt2.y << endl;

			_x1 = int(x0 + 1000 * (-b));
			_y1 = int(y0 + 1000 * (a));
			_x2 = int(x0 - 1000 * (-b));
			_y2 = int(y0 - 1000 * (a));

			x1 += _x1;
			y1 += _y1;

			x2 += _x2;
			y2 += _y2;

			break;
		};
	}
	if (count != 0) {
		p1->x = x1 / 2; p1->y = y1 / 2;
		p2->x = x2 / 2; p2->y = y2 / 2;

		left_error_count = 0;
		return false;
	}
	return true;
}

bool Lane_Detector::hough_right(Mat& img, Point* p1, Point* p2) {
	vector<Vec2f> linesR;

	int count = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
	int threshold = 40;

	for (int i = 10; i > 0; i--) {
		HoughLines(img, linesR, 1, CV_PI / 180, threshold, 0, 0, CV_PI / 2, CV_PI);
		int clusterCount = 2;
		Mat h_points = Mat(linesR.size(), 1, CV_32FC2);
		Mat labels, centers;
		if (linesR.size() > 1) {
			for (size_t i = 0; i < linesR.size(); i++) {
				count++;
				float rho = linesR[i][0];
				float theta = linesR[i][1];
				double a = cos(theta), b = sin(theta);
				double x0 = a * rho, y0 = b * rho;
				// cout << "x0, y0 : " << rho << ' ' << theta << endl;
				h_points.at<Point2f>(i, 0) = Point2f(rho, (float)(theta * 100));
			}
			// kmeans(h_points, clusterCount, labels,
			// 	TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
			// 	3, KMEANS_RANDOM_CENTERS, centers);

			Point mypt1 = centers.at<Point2f>(0, 0);

			float rho = mypt1.x;
			float theta = (float)mypt1.y / 100;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;

			// cout << "pt : " << mypt1.x << ' ' << mypt1.y << endl;

			int _x1 = int(x0 + 1000 * (-b));
			int _y1 = int(y0 + 1000 * (a));
			int _x2 = int(x0 - 1000 * (-b));
			int _y2 = int(y0 - 1000 * (a));

			x1 += _x1;
			y1 += _y1;

			x2 += _x2;
			y2 += _y2;

			Point mypt2 = centers.at<Point2f>(1, 0);

			rho = mypt2.x;
			theta = (float)mypt2.y / 100;
			a = cos(theta), b = sin(theta);
			x0 = a * rho, y0 = b * rho;

			// cout << "pt : " << mypt2.x << ' ' << mypt2.y << endl;

			_x1 = int(x0 + 1000 * (-b));
			_y1 = int(y0 + 1000 * (a));
			_x2 = int(x0 - 1000 * (-b));
			_y2 = int(y0 - 1000 * (a));

			x1 += _x1;
			y1 += _y1;

			x2 += _x2;
			y2 += _y2;

			break;
		};
	}
	if (count != 0) {
		p1->x = x1 / 2; p1->y = y1 / 2;
		p2->x = x2 / 2; p2->y = y2 / 2;

		right_error_count = 0;
		return false;
	}
	return true;
}

int Lane_Detector::argmax(int* arr, int size) {
	int max = 0;
	int max_index = -1;
	for(int i = 0; i < size; i++) {
		if(arr[i] > max) {
			max = arr[i];
			max_index = i;
		}
	} 
	return max_index;
}
#endif

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
#include <deque>
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
	vector<double> polyfit(int size, vector<int> vec_x, vector<int> vec_y, int degree);
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
	imremapped = imremapped(Rect(0, 0, 200, 140));
	cvtColor(originImg, imgray, CV_BGR2GRAY);
    inverse_perspective_mapping(DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, imgray.data, ipm_table, imremapped.data);
	Canny(imremapped, cannyImg, 100, 300);
	//imshow("origin", imremapped);
	//waitKey(0);
	//imshow("canny", cannyImg);
	Mat roi_l = cannyImg(Rect(0, 0, 100, 140));
	Mat roi_r = cannyImg(Rect(100, 0, 100, 140));
	Mat dilated_l, dilated_r;
	
	morphologyEx(roi_l, dilated_l, MORPH_CLOSE, Mat(9,9, CV_8U, Scalar(1)));
	morphologyEx(roi_r, dilated_r, MORPH_CLOSE, Mat(9,9, CV_8U, Scalar(1)));
	
	//imshow("closed_l", dilated_l);
	//imshow("closed_r", dilated_r);
	int histogram_l[100] = {0};
	int histogram_r[100] = {0};
	vector<int> nonzero_l_x, nonzero_l_y, nonzero_r_x, nonzero_r_y; 
	for(int i = 0; i < 140; i++) {
		for(int j = 0; j < 100; j++){
			if(dilated_l.data[i * dilated_l.step + j] != 0){
				nonzero_l_y.push_back(i);
				nonzero_l_x.push_back(j);
			}
			if(dilated_r.data[i * dilated_l.step + j] != 0){
				nonzero_r_y.push_back(i);
				nonzero_r_x.push_back(j);
			}
		}
	}
	printf("=============left===============\n");
	vector<double> poly_left = polyfit(nonzero_l_x.size(), nonzero_l_x, nonzero_l_y, 1);
	// printf("============right===============\n");
	// vector<double> poly_right = polyfit(nonzero_r_x.size(), nonzero_r_x, nonzero_r_y, 1);

	for(int i = 0; i < 100; i++){
		double y = (poly_left[0] * pow(i, 0) + poly_left[1] * pow(i,1));
		printf("%f ",y);
		circle(dilated_l, Point(i, y), 5, Scalar(255,255,255));
	}
	printf("\n");
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
	Mat result;
	hconcat(dilated_l, dilated_r, result);
	imshow("Result", result);
	// int i = 200;
	// int j = 180;
	// while (j >= 0){3
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
	// //imshow("originImg", imremapped);
	waitKey(0);
}
vector<double> Lane_Detector::polyfit(int size, vector<int> vec_x, vector<int> vec_y, int degree){
	int n = degree;
	double X[2*n+1];
	
	for(int i = 0; i<2*n+1; i++){
		X[i]=0;
		for(int j = 0; j<size; j++){
			X[i]=X[i]+pow(vec_x[j]/10, i);
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
			Y[i]=Y[i]+pow(vec_x[j]/10, i)*vec_y[j]/10;
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
		a[i]=B[i][n];
		for(int j = 0; j<n; j++){
			if(j != i){
				a[i] = a[i] - B[i][j] * a[j];
			}
		}
		a[i]=a[i]/B[i][i];
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
			
			kmeans(h_points, clusterCount, labels,
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
				3, KMEANS_RANDOM_CENTERS, centers);

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
			kmeans(h_points, clusterCount, labels,
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1.0),
				3, KMEANS_RANDOM_CENTERS, centers);

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
	int max_index = 0;
	for(int i = 0; i < size; i++) {
		if(arr[i] >= max) {
			max = arr[i];
			max_index = i;
		}
	} 
	return max_index;
}
#endif

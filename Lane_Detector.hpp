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
	Point p1, p2, p3, p4;
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
	VideoWriter outputVideo;

	double prev_slope_l, prev_slope_r;
	double prev_intercept_l, prev_intercept_r;
	Point prev_p_l, prev_p_r;
	bool isinit = true;
	int error_count_l, error_count_r;
	void base_ROI(Mat& img, Mat& img_ROI);
	void v_roi(Mat& img, Mat& img_ROI, const Point& p1, const Point& p2);
	void region_of_interest_L(Mat& img, Mat& img_ROI);
	void region_of_interest_R(Mat& img, Mat& img_ROI);
	bool hough_left(Mat& img, Point* p1, Point* p2);
	bool hough_right(Mat& img, Point* p1, Point* p2);
	float get_slope(const Point& p1, const Point& p2);
	int position(const Point P1, const Point P2);
	int argmax(int* arr, int size);
	int argmax_r(int* arr, int size);
	vector<double> polyfit(int size, vector<Point> vec_p, int degree);
	vector<int> filter_line(vector<int> vec_nonzero);
	vector<double> left_lane;
	vector<double> right_lane;
public:
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
	vector<double> get_left_lane();
	vector<double> get_right_lane();
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
vector<double> Lane_Detector::get_left_lane(){
	return left_lane;
}
vector<double> Lane_Detector::get_right_lane(){
	return right_lane;
}
void Lane_Detector::init() {
	string path = "output.avi";
	struct tm* datetime;
	time_t t;
	t = time(NULL);
	datetime = localtime(&t);
	string s_t = path;

	mask = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

	left_error = true;
	right_error = true;
	left_length = 0;
	right_length = 0;
	left_error_count = 0;
	right_error_count = 0;
	prev_slope_l = 0;
	prev_slope_r = 0;
	prev_p_l = Point(0, 0);
	prev_p_r = Point(0, 0);
	error_count_l = 0;
	error_count_r = 0;
	vanishing_point_x = 320;
	vanishing_point_y = 235;
	left_lane = vector<double>(2);
	right_lane = vector<double>(2);
	
	//outputVideo.open(s_t, VideoWriter::fourcc('X', 'V', 'I', 'D'), 10, Size(720, 120), true);
	outputVideo.open(s_t, VideoWriter::fourcc('D', 'I', 'V', 'X'), 10, Size(720, 120), true);
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
	Mat dilatedImg;
	morphologyEx(cannyImg, dilatedImg, MORPH_CLOSE, Mat(9,9, CV_8U, Scalar(1)));
	Mat roi_l = dilatedImg.clone();
	Mat roi_r = dilatedImg.clone();
	
	for(int i = 0; i < 120; i++){
		for(int j = 70; j < 140; j++){
			roi_l.data[i * roi_l.step + j] = 0;
		}
		for(int j = 0; j < 70; j++){
			roi_r.data[i * roi_r.step + j] = 0;
		}
	}
	
	
	// imshow("closed_l", dilated_l);
	// imshow("closed_r", dilated_r);
	int histogram_l[4][140] = {0};
	int histogram_r[4][140] = {0};
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 30; j++){
			for(int k = 0; k < 140; k++){
				histogram_l[i][k] += roi_l.data[(i * 30 + j) * roi_l.step + k];
				histogram_r[i][k] += roi_r.data[(i * 30 + j) * roi_r.step + k];
			}
		}
	}
	int max_arg_l[4], max_arg_r[4];
	
	cvtColor(imremapped, imremapped, COLOR_GRAY2BGRA);
	int zero_count_l = 0, zero_count_r = 0;
	vector<Point> points_l, points_r;
	vector<double> poly_left, poly_right;
	for(int i = 0; i < 4; i++){
		// for(int j = 1; j <= 22; j+=2){
		// 	line(imremapped, Point(7*(j), i*30 + 30), Point(7*(j+1), i*30 + 30), COLOR_RED, 1, CV_AA);
		// }
		max_arg_l[i] = argmax_r(histogram_l[i], 140);
		max_arg_r[i] = argmax(histogram_r[i], 140);
		
		if(max_arg_l[i] != -1){
			double d = abs(max_arg_l[i] - prev_slope_l * (i * 30 + 15) - prev_intercept_l)/sqrt(1+pow(prev_slope_l, 2));
			//printf("left point distance : %f\n", d);
			if(isinit || d < 25 || error_count_l < 5)
				points_l.push_back(Point(i * 30 + 15, max_arg_l[i]));
		}
		if(max_arg_r[i] != -1){
			double d = abs(max_arg_r[i] - prev_slope_r * (i * 30 + 15) - prev_intercept_r)/sqrt(1+pow(prev_slope_r, 2));
			//printf("right point distance : %f\n", d);
			if(isinit || d < 25 || error_count_r < 5)
				points_r.push_back(Point(i * 30 + 15, max_arg_r[i]));
		}
	}	
	
	
	Mat forcurve;
	forcurve = imremapped.clone();
	int degree_l = 1, degree_r = 1;
	//printf("prev left: %fx: \n",prev_slope_l);
	//printf("prev right: %fx: \n",prev_slope_r);
	//printf("left: ");
	poly_left = polyfit(points_l.size(), points_l, degree_l);
	//printf("right: ");
	poly_right = polyfit(points_r.size(), points_r, degree_r);
	
	if(isnan(poly_left[1])){
		error_count_l++;
	}
	else{
		error_count_l = 0;
	}
	if(isnan(poly_right[1])){
		error_count_r++;
	}
	else{
		error_count_r = 0;
	}
	// printf("error count left : %d\n", error_count_l);
	// printf("error count right : %d\n", error_count_r);
	if(abs(poly_left[1]) < 0.04f) {
		poly_left[1] = 0.f;
		if(abs(poly_right[1]) > 0.04f && abs(poly_right[1]) < 0.1f){
			poly_right[1] = 0.f;
		}
	}
	else if(abs(prev_slope_l - poly_left[1]) > 0.5){
		poly_left[1] = prev_slope_l;
		poly_left[0] = prev_intercept_l;
		//printf("new left: %fx\n", poly_left[1]);
	}
	if(abs(poly_right[1]) < 0.04f) {
		poly_right[1] = 0.f;
		if(abs(poly_left[1]) > 0.04 && abs(poly_left[1]) < 0.1f){
			poly_left[1] = 0.f;
		}
	}
	else if(abs(prev_slope_r - poly_right[1]) > 0.5){
		poly_right[1] = prev_slope_r;
		poly_right[0] = prev_intercept_r;
		//printf("new right: %fx\n", poly_right[1]);
	}
	
	if(isnan(poly_left[1]) || isnan(poly_right[1]) ||  prev_slope_l * poly_left[1] < 0 || poly_left[1] * poly_right[1] < -0.1) {
		if(isnan(poly_left[1]) && isnan(poly_right[1])){
			poly_left[1] = prev_slope_l;
			poly_left[0] = prev_intercept_l;
			poly_right[1] = prev_slope_r;
			poly_right[0] = prev_intercept_r;
		}
		else if(isnan(poly_left[1]) || prev_slope_l * poly_left[1] < 0.0f) {
			poly_left[1] = poly_right[1];
			poly_left[0] = poly_right[0] - 100*sqrt(1+pow(poly_right[1],2));
			//printf("new left: %fx^1 + %fx^0\n", poly_left[1], poly_left[0]);
		}
		else if(isnan(poly_right[1]) || prev_slope_r * poly_right[1] < 0.f){
			poly_right[1] = poly_left[1];
			poly_right[0] = poly_left[0] + 100*sqrt(1+pow(poly_left[1],2));
			//printf("new right: %fx^1 + %fx^0\n", poly_right[1], poly_right[0]);
		}
	}
	double distance = (poly_right[0] - poly_left[0]) / sqrt(1 + pow(poly_left[1], 2));
	// printf("distance : %f\n", distance);
	if (abs(distance) < 90) {
		if((distance < 90 && distance > 0)) {
			poly_left[1] = poly_right[1];
			poly_left[0] = poly_right[0] - 100*sqrt(1+pow(poly_right[1],2));
			// printf("new left: %fx^1 + %fx^0\n", poly_left[1], poly_left[0]);
		}
		else if((distance < 0 && distance > -90)){
			poly_right[1] = poly_left[1];
			poly_right[0] = poly_left[0] + 100*sqrt(1+pow(poly_left[1],2));
			// printf("new right: %fx^1 + %fx^0\n", poly_right[1], poly_right[0]);
		}
	}
	if (abs(poly_right[1]-poly_left[1]) > 0.1){
		double diff_left = abs(poly_left[1] - prev_slope_l);
		double diff_right = abs(poly_right[1] - prev_slope_r);
		if(diff_left > diff_right){
			poly_left[1] = poly_right[1];
		}
		else {
			poly_right[1] = poly_left[1];
		}
	}
	for(int i = 0; i < 120; i++){
		int x = 0;
		for(int j = 0; j <= degree_l; j++)
			x += poly_left[j] * pow(i, j);
		if(x >= 0){
			circle(forcurve, Point(x+2, i), 3, Scalar(255, 255, 0), -1);
			prev_p_l = Point(x, i);
		}
	}
	for(int i = 0; i < 120; i++){
		int x = 0;
		for(int j = 0; j <= degree_r; j++)
			x += poly_right[j] * pow(i, j);
		if(x >= 0){
			circle(forcurve, Point(x+2, i), 3, Scalar(0, 255, 255), -1);
			prev_p_r = Point(x, i);
		}
	}
	prev_slope_l = poly_left[1];
	prev_intercept_l = poly_left[0];
	prev_slope_r = poly_right[1];
	prev_intercept_r = poly_right[0];

	addWeighted(forcurve, 0.3, imremapped, 0.7, 0.0, imremapped);
	resize(originImg, originImg, Size(160, 120), 0, 0, CV_INTER_NN);
	cvtColor(originImg, originImg, CV_BGR2BGRA);
	Mat result, result2;
	cvtColor(cannyImg, cannyImg, CV_GRAY2BGRA);
	cvtColor(roi_l, roi_l, CV_GRAY2BGRA);
	cvtColor(roi_r, roi_r, CV_GRAY2BGRA);
	hconcat(cannyImg, roi_l, roi_l);
	hconcat(roi_l, roi_r, result);
	 //cvtColor(result, result, CV_GRAY2BGRA);
	hconcat(result, imremapped, result2);
	hconcat(originImg, result2, result2);
	//imshow("Img", imremapped);
	imshow("Img", result2);
	if(isinit){
		isinit = false;
	}
	cvtColor(result2, result2, CV_BGRA2BGR);
	//outputVideo << result2;
	left_lane[0] = poly_left[0];
	left_lane[1] = poly_left[1];
	right_lane[0] = poly_right[0];
	right_lane[1] = poly_right[1];
	if(waitKey(10)==0){
		return;
	}
	
}
vector<double> Lane_Detector::polyfit(int size, vector<Point> vec_p, int degree){
	int n = degree;
	double X[2*n+1];
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
		// printf("%fx^%d", a[i], i);
		if(i != 0){
			// printf(" + ");
		}
		else{
			// printf("\n");
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
int Lane_Detector::argmax_r(int* arr, int size) {
	int max = 0;
	int max_index = -1;
	for(int i = size-1; i >= 0; i--) {
		if(arr[i] > max) {
			max = arr[i];
			max_index = i;
		}
	} 
	return max_index;
}
#endif

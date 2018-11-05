#pragma once
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;

class Filter
{
public:
	Filter();
	~Filter();

	///Get contours, face and eyes and draw them in colors
	Mat FirstFilter(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);

	///Put mask on face
	Mat SecondFilter(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade, Mat image,float scale = 1.0);

	///Get only skin
	Mat ThirdFilter(Mat frame);
	
	
	

private:
	void DetectFaceAndEyes(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);
	void PutMask(const Mat &background, const Mat &foreground, Mat &output, Point2i location);

	vector<Rect> faces;
	vector<Rect> eyes;

	int Y_MIN;
	int Y_MAX;
	int Cr_MIN;
	int Cr_MAX;
	int Cb_MIN;
	int Cb_MAX;

	
};


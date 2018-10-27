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

	Mat FirstFilter(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);
	Mat SecondFilter(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade, Mat image,float scale = 1.0);
	
	
	

private:
	void DetectFaceAndEyes(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);
	Mat PutMask(Mat src, Mat mask, Point center, Size face_size);

	vector<Rect> faces;
	vector<Rect> eyes;
	vector<Point> orientation;

	
};


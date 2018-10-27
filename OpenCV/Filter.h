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

	Mat DetectAndDisplayFirstFilter(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);
	
	
	

private:
	void DetectFaceAndEyes(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);

	vector<Rect> faces;
	vector<Rect> eyes;
};


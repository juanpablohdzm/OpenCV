#pragma once
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect.hpp"

using namespace std;
using namespace cv;



class Video
{
public:
	Video();
	~Video();

	int LoadCascades();
	int OpenCapture();
	Mat ReadFrame();


	CascadeClassifier GetFace_cascade() const { return face_cascade; }
	CascadeClassifier GetEyes_cascade() const { return eyes_cascade; }
	
private:
	String face_cascade_name;
	String eyes_cascade_name;

	VideoCapture capture;
	Mat frame;

	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;
};


#include "Video.h"



Video::Video()
{
	face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml";
	eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
}


Video::~Video()
{
}

int Video::LoadCascades()
{
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
}

int Video::OpenCapture()
{
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }
	return 0;
	
}

cv::Mat Video::ReadFrame()
{
	capture.read(frame);
	return frame;
	
}

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include "Video.h"
#include "Filter.h"

using namespace std;
using namespace cv;

Video video;
Filter filter;

String window_name = "Capture - Face detection";



int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|haarcascades/haarcascade_frontalface_alt.xml|}"
		"{eyes_cascade|haarcascades/haarcascade_eye_tree_eyeglasses.xml|}");
	cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n";
	parser.printMessage();
	
	
	//-- 1. Load the cascades
	video.LoadCascades();
	video.OpenCapture();
	
	//-- 2. Read the video stream
	Mat frame;
	do 
	{
		frame= video.ReadFrame();
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			return -1;
		}
		//-- 3. Apply the classifier to the frame
		Mat drawing =filter.DetectAndDisplayFirstFilter(frame, video.GetFace_cascade(), video.GetEyes_cascade());
		imshow(window_name, drawing);
		char c = (char)waitKey(10);
		if (c == 27) { break; } // escape

	} while (true);
	
	return 0;
}


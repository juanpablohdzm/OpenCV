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
int choice = 0;

Mat3b canvas;
Rect button;
String buttonText("Click me!");

String window_name = "Capture - Face detection";


void OnButtonClick(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (button.contains(Point(x, y)))
		{
			choice++;
			//rectangle(canvas(button), button, Scalar(0, 0, 255), 2);
		}
	}
	imshow(window_name, canvas);
}

int main(int argc, const char** argv)
{
	
	//-- Load the cascades and open capture
	video.LoadCascades();
	video.OpenCapture();
	
	//-- Load image
	Mat image = imread("Images/mask.png", IMREAD_UNCHANGED);
	if (image.empty()) { cout << "Error image not found" << endl; return -1; }
	//-- Read the video stream
	Mat frame;

	do 
	{
		frame= video.ReadFrame();

		//Create UI
		button = Rect(0, 0, frame.cols, 50);
		canvas = Mat3b(frame.rows + button.height, frame.cols, Vec3b(0, 0, 0));
		canvas(button) = Vec3b(200, 200, 200);
		putText(canvas(button),buttonText, Point(button.width*0.35, button.height*0.7), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));
		

		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			return -1;
		}
		//-- 3. Apply the classifier to the frame
		Mat drawing;
		switch (choice%3)
		{
		case 0:
			drawing = filter.FirstFilter(frame, video.GetFace_cascade(), video.GetEyes_cascade());
			break;
		case 1:
			drawing = filter.SecondFilter(frame, video.GetFace_cascade(), video.GetEyes_cascade(),image);
			break;
		case 2:
			drawing = filter.ThirdFilter(frame);
		default:
			break;
		}
		drawing.copyTo(canvas(Rect(0, button.height, frame.cols, frame.rows)));
		imshow(window_name, canvas);
		setMouseCallback(window_name, OnButtonClick);
		char c = (char)waitKey(10);
		if (c == 27) { break; } // escape

	} while (true);
	return 0;
}




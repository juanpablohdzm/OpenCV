#include "Filter.h"



Filter::Filter()
{
}


Filter::~Filter()
{
}

void Filter::DetectFaceAndEyes(Mat frame,CascadeClassifier face_cascade, CascadeClassifier eye_cascade)
{
	
	equalizeHist(frame, frame);
	//-- Detect faces
	face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		//-- In each face, detect eyes
		Mat faceROI = frame(faces[i]);
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	}
}

Mat Filter::DetectAndDisplayFirstFilter(Mat frame,CascadeClassifier face_cascade,CascadeClassifier eye_cascade)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	DetectFaceAndEyes(frame_gray, face_cascade, eye_cascade);
	
	Mat canny_output;
	RNG rng(12345);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	//Draw circles for face and eyes. 
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame_gray, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
						
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame_gray, eye_center, radius, Scalar(0, 0, 0), FILLED, 5, 0);

		}

		if (eyes.size() == 2) {
			Point ojo1(faces[i].x + eyes[0].x + eyes[0].width / 2, faces[i].y + eyes[0].y + eyes[0].height / 2);
			Point ojo2(faces[i].x + eyes[1].x + eyes[1].width / 2, faces[i].y + eyes[1].y + eyes[1].height / 2);
			line(frame_gray, ojo1, ojo2,
				Scalar(0, 0, 0),
				8,
				1);
		}

	}
	
	
	blur(frame_gray, frame_gray, Size(3, 3));
	Canny(frame_gray, canny_output, 100, 100 * 2, 3);
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Moments> mu(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}
	vector<Point2f> mc(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(static_cast<float>(mu[i].m10 / mu[i].m00), static_cast<float>(mu[i].m01 / mu[i].m00));
	}
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 4, color, -1, 8, 0);
	}


	return drawing;
	
}


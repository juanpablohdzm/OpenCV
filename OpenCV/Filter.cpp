#include "Filter.h"





Filter::Filter()
{

	Y_MIN = 0;
	Y_MAX = 255;
	Cr_MIN = 133;
	Cr_MAX = 173;
	Cb_MIN = 77;
	Cb_MAX = 127;
}


Filter::~Filter()
{
}

void Filter::DetectFaceAndEyes(Mat frame,CascadeClassifier face_cascade, CascadeClassifier eye_cascade)
{
	
	equalizeHist(frame, frame);
	//-- Detect faces
	face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
	for (size_t i = 0; i < faces.size(); i++)
	{
		//-- In each face, detect eyes
		Mat faceROI = frame(faces[i]);
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	}
}

void Filter::PutMask(const Mat &background, const Mat &foreground,Mat &output, Point2i location)
{

	background.copyTo(output);


	// start at the row indicated by location, or at row 0 if location.y is negative.
	for (int y = max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y; // because of the translation

		// we are done of we have processed all rows of the foreground image.
		if (fY >= foreground.rows)
			break;

		// start at the column indicated by location, 

		// or at column 0 if location.x is negative.
		for (int x = max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; // because of the translation.

			// we are done with this row if the column is outside of the foreground image.
			if (fX >= foreground.cols)
				break;

			// determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
			double opacity =((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3])/ 255.;


			// and now combine the background and foreground pixel, using the opacity, 

			// but only if opacity > 0.
			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx =
					foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx =
					background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] =
					backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}
}

Mat Filter::FirstFilter(Mat frame,CascadeClassifier face_cascade,CascadeClassifier eye_cascade)
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

Mat Filter::SecondFilter(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade, Mat image,float scale)
{	
	Mat frame_gray;
	Mat drawing;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	DetectFaceAndEyes(frame_gray, face_cascade, eye_cascade);
	
	
	
	
	for (size_t i = 0; i < faces.size(); ++i)
	{
		resize(image, image, Size(faces[i].width, faces[i].height));
		Point2i center = Point2i(faces[i].x, faces[i].y);
		PutMask(frame, image, drawing, center);
		frame = drawing;
	}
	return frame;

}

Mat Filter::ThirdFilter(Mat frame)
{
	Mat skin, dstImage;
	cvtColor(frame, skin, COLOR_BGR2YCrCb);
	inRange(skin, Scalar(Y_MIN, Cr_MIN, Cb_MIN), Scalar(Y_MAX, Cr_MAX, Cb_MAX), skin);

	frame.copyTo(dstImage, skin);
	return dstImage;
}


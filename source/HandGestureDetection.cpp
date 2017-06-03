
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>

using namespace cv;
using namespace std;

int hmin = 0,	hmax = 30;
int smin = 30,	smax = 175;
int vmin = 90,	vmax = 255;

////////////////////////////////////////////////////////////////
bool detect_hand(Mat image, Mat background, Mat& mask);
bool detect_gesture(Mat image);
void display_result(String wndname, Mat image);
//------------------------------------------------------------//
int main(int argc, char **argv)
{
	char fname[256];
	if (argc > 1)
		sprintf(fname, "%s", argv[1]);
	else
		// for debug ==============
		//sprintf(fname, "%s", "..\\videos\\Experiment5.mov");
		sprintf(fname, "%s", "");

	VideoCapture cap;
	bool bopened;
	if (fname[0] != 0)
		bopened = cap.open(fname);
	else
		bopened = cap.open(0); // from webcam

	if (!bopened) {
		cerr << "Video stream read failed." << endl;
		return 0;
	}

	Mat frame, image, bckgrd, handmask;
	int nW, nH;
	Rect rtHand;
	
	nW = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	nH = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	rtHand = Rect(10, 10, nW - 20, nH - 20);

	namedWindow("video", WINDOW_NORMAL);
	resizeWindow("video", nW, nH);

	int nstart = 0;
	while (true) {
		if (!cap.read(frame)) break;

		flip(frame, frame, 1);

		rectangle(frame, rtHand, Scalar(0, 0, 255));

		// user interface
		char key = waitKey(25);
		if (key == 'q' || key == 27/*ESC*/) break;
		if (key == 'p') {
			nstart = 0; continue;
		}
		if (key == 's') {
			nstart = 1;
			display_result("video", frame);
			continue;
		}

		if (nstart == 0) {
			string str = "Press 's' to start gesture detection.";
			putText(frame, str, Point(nW / 2 - str.length() * 12 / 2, nH - 50),
				FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 0), 2, LINE_8);

			display_result("video", frame);
			continue;
		}
		if (nstart < 3) {
			nstart++;
			display_result("video", frame);
			continue;
		}

		// grab background image
		if (nstart == 3) {
			filter2D(frame(rtHand), bckgrd, CV_8U, Mat::ones(5, 5, CV_32FC1) / 25);
			nstart++;
			display_result("video", frame);
			continue;
		}

		// detect hand gesture
		filter2D(frame(rtHand), image, CV_8U, Mat::ones(5, 5, CV_32FC1) / 25);

		if (!detect_hand(image, bckgrd, handmask)) {
			display_result("video", frame);
			continue;
		}

		display_result("video", frame);
	
	}

	cap.release();
	
	destroyAllWindows();

	return 0;
}

bool detect_hand(Mat image, Mat background, Mat& mask)
{
	Mat bwim;

	absdiff(image, background, mask);
	filter2D(mask, mask, CV_8U, Mat::ones(2, 2, CV_32FC1));

	imshow("Mask", mask);

	return true;
}

bool detect_gesture(Mat image)
{
	return true;
}

void display_result(String wndname, Mat image)
{
	imshow(wndname, image);
}

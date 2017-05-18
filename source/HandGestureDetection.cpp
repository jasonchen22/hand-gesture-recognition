
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////
bool detect_hand(Mat image);
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

	Mat frame;
	int nW, nH;
	
	nW = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	nH = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

	namedWindow("video", WINDOW_NORMAL);
	resizeWindow("video", nW, nH);

	namedWindow("video", WINDOW_NORMAL);
	//createTrackbar("color controller", "video", )

	while (true) {
		if (!cap.read(frame)) break;

		flip(frame, frame, 1);

		display_result("video", frame);
		
		char key = waitKey(25);
		if (key == 'q' || key == 27/*ESC*/) break;
	}

	cap.release();

	return 1;
}

bool detect_hand(Mat image)
{
	Mat imhsv;

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

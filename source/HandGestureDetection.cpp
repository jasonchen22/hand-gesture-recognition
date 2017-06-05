
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <time.h>
#include <fstream>

using namespace cv;
using namespace std;

int ncount;
////////////////////////////////////////////////////////////////
bool detect_hand(Mat image, Mat background, vector<Point>& handcontour);
int detect_gesture(Mat image, vector<Point> handcontour, Point& indexpos);
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
	Point indexpos;
	vector<Point> handcontour;
	
	nW = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	nH = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	rtHand = Rect(nW / 2, 0, nW / 2, nH);

	namedWindow("video", WINDOW_NORMAL);
	resizeWindow("video", nW, nH);

	ncount = 0; // for debug
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
			filter2D(frame(rtHand), bckgrd, CV_8U, Mat::ones(3, 3, CV_32FC1) / 9);
			cvtColor(bckgrd, bckgrd, COLOR_BGR2GRAY);
			nstart++;
			display_result("video", frame);
			continue;
		}

		// detect hand gesture
		filter2D(frame(rtHand), image, CV_8U, Mat::ones(3, 3, CV_32FC1) / 9);
		cvtColor(image, image, COLOR_BGR2GRAY);

		if (!detect_hand(image, bckgrd, handcontour)) {
			display_result("video", frame);
			continue;
		}

		int igesture = detect_gesture(image, handcontour, indexpos);


		display_result("video", frame);
	
	}

	cap.release();
	
	destroyAllWindows();

	return 0;
}

bool detect_hand(Mat image, Mat background, vector<Point>& handcontour)
{
	Mat bwim;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

	handcontour = vector<Point>(0);

	absdiff(image, background, bwim);
	dilate(bwim, bwim, kernel);
	imshow("Mask1", bwim);

	threshold(bwim, bwim, 20, 255, CV_8U);

	morphologyEx(bwim, bwim, MORPH_CLOSE, kernel);

	Rect rt;
	vector<vector<Point> > contours;
	vector<Vec4i > hierarchy;
	double armx = 0;
	int k = -1;

	findContours(bwim, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++) {
		double ar = contourArea(contours[i]);
		if (ar <= armx) continue;

		armx = ar;
		k = i;
	}

	if (armx < 50 * 50) return false;
	rt = boundingRect(contours[k]);
	if (rt.width < 100 && rt.height < 100) return false;

	handcontour = contours[k];

	// for debug
	Mat ima = Mat::zeros(image.size(), CV_8UC1);
	drawContours(ima, contours, k, Scalar(255), -1);
	imshow("Mask", ima);
	//char str[256];
	//sprintf(str, "5\\%d.jpg", ncount);
	//imwrite(str, ima);
	//ncount++;

	return true;
}

double angle_2vectors(Point2f vec1, Point2f vec2);
int detect_gesture(Mat image, vector<Point> handcontour, Point& indexpos)
{
	int i, i1, i2;
	Point vec1, vec2, pt, pt1, pt2;
	vector<Point> approxpoly;
	
	approxPolyDP(handcontour, approxpoly, 30, true);

	vector<int> yy(3), ind(3);
	yy[0] = image.rows; ind[0] = -1;
	yy[1] = image.rows; ind[1] = -1;
	yy[2] = image.rows; ind[2] = -1;
	for (i = 0; i < approxpoly.size(); i++) {
		if (approxpoly[0].y < yy[0]) {
			yy[2] = yy[1]; ind[2] = ind[1];
			yy[1] = yy[0]; ind[1] = ind[0];
			yy[0] = approxpoly[0].y; ind[0] = i;
		}
		else if (approxpoly[0].y < yy[1]) {
			yy[2] = yy[1]; ind[2] = ind[1];
			yy[1] = approxpoly[0].y; ind[1] = i;
		}
		else if (approxpoly[0].y < yy[2]) {
			yy[2] = approxpoly[0].y; ind[2] = i;
		}
	}
	if (ind[0] == -1) return -1;

	int dthr = 30;
	for (i = 0; i < 3; i++) {
		pt = approxpoly[ind[i]];
		i1 = ind[i] - 1;
		if (i1 < 0) i1 = approxpoly.size() - 1;
		pt1 = approxpoly[i1];
		vec1 = pt1 - pt;
		if (vec1.x*vec1.x + vec1.y*vec1.y < dthr*dthr) {
			i1--; pt1 = approxpoly[i1];
			vec1 = pt1 - pt;
			if (vec1.x*vec1.x + vec1.y*vec1.y < dthr*dthr) continue;
		}

		i2 = ind[i] + 1;
		if (i2 >= approxpoly.size()) i2 = 0;
		pt2 = approxpoly[i2];
		vec2 = pt2 - pt;
		if (vec2.x*vec2.x + vec2.y*vec2.y < dthr*dthr) {
			i2++; pt2 = approxpoly[i2];
			vec2 = pt2 - pt;
			if (vec2.x*vec2.x + vec2.y*vec2.y < dthr*dthr) continue;
		}

		double al = angle_2vectors(vec1, vec2);
		if (al > 45) continue;
	}

	polylines(image, approxpoly, true, Scalar(128));
	imshow("approx", image);

	return true;
}

void display_result(String wndname, Mat image)
{
	imshow(wndname, image);
}

double angle_2vectors(Point2f vec1, Point2f vec2)
{
	double al1 = (atan2(vec1.y, vec1.x) * 180 / 3.141592);
	double al2 = (atan2(vec2.y, vec2.x) * 180 / 3.141592);

	double al = abs(al1 - al2);
	al = al < 180 ? al : 360 - al;

	return al;
}
double cross_correlation(Mat img1, Mat img2, bool isbinary)
{
	double corr;

	Scalar img1_avg, img2_avg;
	if (isbinary) {
		// in the case with binary image
		img1_avg = Scalar(128, 0, 0);
		img2_avg = Scalar(128, 0, 0);
	}
	else {
		// in the case with gray image
		img1_avg = mean(img1);
		img2_avg = mean(img2);
	}

	double sum_img1_img2 = 0;
	double sum_img1_2 = 0;
	double sum_img2_2 = 0;

	for (int m = 0; m<img1.rows; ++m)
	{
		for (int n = 0; n<img1.cols; ++n)
		{
			sum_img1_img2 += (img1.at<uchar>(m, n) - img1_avg.val[0])*(img2.at<uchar>(m, n) - img2_avg.val[0]);
			sum_img1_2 += (img1.at<uchar>(m, n) - img1_avg.val[0])*(img1.at<uchar>(m, n) - img1_avg.val[0]);
			sum_img2_2 += (img2.at<uchar>(m, n) - img2_avg.val[0])*(img2.at<uchar>(m, n) - img2_avg.val[0]);
		}
	}

	if (sum_img1_2 == 0 || sum_img2_2 == 0) return -1;

	corr = sum_img1_img2 / sqrt(sum_img1_2*sum_img2_2);

	return corr;
}

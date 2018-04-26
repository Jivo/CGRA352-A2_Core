
// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "nnf.hpp"
#include "reconstruction.hpp"
#include "gauss_pyr.hpp"

#include "globalVars.cpp"


using namespace cv;
using namespace std;

Rect findMaskBounds(const Mat& mask);
void swapPatch(const Mat& source, Mat& target, const Mat& mask, Point translation);
Mat addBorder(const Mat&m);

int main( int argc, char** argv ) {
	if( argc != 3) {
		cout << "Wrong number of arguments: " << argc<<" REQUIRED: 3"<<endl;
		return -1;}
	
	Mat source, target;
	source = imread(argv[1], CV_LOAD_IMAGE_COLOR); 
	target = imread(argv[2], CV_LOAD_IMAGE_COLOR);

	if(!source.data || !target.data) {
		cout << "Could not open or find the image" << std::endl;
		return -1;}

	Mat sourceBorder = addBorder(source);
	Mat targetBorder = addBorder(target);
	
	/****************************
	           START
	*****************************/

	//Algorithm
	Mat nnf(target.rows, target.cols, CV_32SC2);
	Mat cost(nnf.rows, nnf.cols, CV_32F);
	initialize(source, target, sourceBorder, targetBorder, nnf, cost);
	iterateNNF(nnf, cost, source, target, sourceBorder, targetBorder);

	Mat reconstr = reconstruct(nnf, source);
	Mat nnfImg = nnf2img(nnf, source.size(), false);

	imwrite("work/res/saves/reconstruction.jpg", reconstr);
	imwrite("work/res/saves/NNF.jpg", nnfImg);

	namedWindow("Reconstruction", WINDOW_AUTOSIZE);
	namedWindow("NNF", WINDOW_AUTOSIZE);
	imshow("Reconstruction", reconstr);
	imshow("NNF", nnfImg);

	/****************************
	           FINISH
	*****************************/


	waitKey(0);
}

/****************************
          SWAP PATCH
*****************************/
void swapPatch(const Mat& source, Mat& target, const Mat& mask, Point off) {
	Rect r1 = findMaskBounds(mask);
	Rect r2 = Rect(r1.x + off.x, r1.y + off.y, r1.width, r1.height);
	Mat retarget = source(r1);
	Mat refill = source(r2);
	retarget.copyTo(target(r2));
	refill.copyTo(target(r1));
}

/****************************
       FIND MASK BOUNDS
*****************************/
Rect findMaskBounds(const Mat& mask) {
	Point p1(-1, -1), p2(-1, -1);
	int i = -1;
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			int val1 = mask.at<Vec3b>(row, col)[0];
			int val2 = mask.at<Vec3b>((mask.rows - 1 - row), (mask.cols - 1 - col))[0];
			if (mask.at<Vec3b>(row, col)[2] > 0 && i == -1) { 
				i++;
			}
			if (val1 > 0 && p1.x == -1) {
				p1 = Point(col, row);}
			if (val2 > 0 && p2.x == -1) {
				p2 = Point(mask.cols - 1 - col,
					mask.rows - 1 - row);}
		}}
	return Rect(p1, Size(p2.x - p1.x, p2.y - p1.y));
}

/****************************
         ADD BORDER
*****************************/
Mat addBorder(const Mat&m) {
	int left = CENTER.x, right = SIZE - 1 - CENTER.x,
		top = CENTER.y, bottom = SIZE - 1 - CENTER.y;

	Mat current = m.clone();
	copyMakeBorder(current, current, top, bottom, left, right, BORDER_CONSTANT);
	return current;
}

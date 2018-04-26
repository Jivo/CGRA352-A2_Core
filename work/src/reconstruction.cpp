#include "reconstruction.hpp"
#include "globalVars.cpp"


/****************************
        RECONSTRUCT
*****************************/
Mat reconstruct(const Mat&nnf, const Mat&source) {
	Mat reconstruction = Mat::zeros(nnf.rows, nnf.cols, source.type());
	for (int row = 0; row < nnf.rows; row++) {
		for (int col = 0; col < nnf.cols; col++) {
			Vec2i offset = nnf.at<Vec2i>(row, col);
			reconstruction.at<Vec3b>(row, col) =
				source.at<Vec3b>(row + offset[0], col + offset[1]);
		}
	}
	
	return reconstruction;
}


/*****************************
NNF TO IMG
******************************/
Mat nnf2img(Mat nnf, Size s, bool absolute) {
	Mat nnfImg(nnf.rows, nnf.cols, CV_8UC3, Scalar(0, 0, 0));
	Rect rect(Point(0, 0), s);
	for (int row = 0; row < nnf.rows; row++) {
		auto inRow = nnf.ptr<Vec2i>(row);
		auto outRow = nnfImg.ptr<Vec3b>(row);

		for (int col = 0; col < nnf.cols; col++) {
			int x = inRow[col][1] + col;
			int y = inRow[col][0] + row;
			if (!rect.contains(Point(x, y))) {
				cout << "Out Of Bounds" << Point(x, y) << " " << s << endl;
				//break;
			}
			outRow[col][2] = int(x*255.0 / s.width);
			outRow[col][1] = int(y*255.0 / s.height);
			outRow[col][0] = 255 - max(outRow[col][2], outRow[col][1]);
		}
	}
	return nnfImg;
}
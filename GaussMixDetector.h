#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

constexpr double defaultT { 0.1 };
constexpr unsigned int defaultHistory { 100U };
constexpr double defaultDeviation { 40.0 };
constexpr double defaultCf { 0.05 };

class GaussMixDetector
{
	// should figure out which parameters can and cannot be changed 'on-the-fly' and
	// add 'set parameter x' function(s) to allow adjusting the model without reinitializing

	static const short K = 3;			// limit number of Gaussians per pixel
	double alpha;						// learning coefficient
	double T;							// background-foreground threshold
	double initDeviation;				// initial 2.5*deviation of all Gaussians
	double Cf;							// portion of FG data

	int fRows, fCols, fChannels;		// frame parameters
	cv::Mat fClone;

	bool firstFrame;					// first step flag
	unsigned int historyLength;			// learning history length

	std::vector <cv::Mat> mean;
	std::vector <cv::Mat> weight;
	std::vector <cv::Mat> deviation;
	cv::Mat currentK;					// current number of Gaussians for each pixel

	static const int CVType = CV_64F;	// type of 'Mat' pixel info
	typedef double ptrType;				// 'Mat' pointer type

public:
	GaussMixDetector();
	explicit GaussMixDetector ( unsigned int _historyLength, double _initDeviation = defaultDeviation, double _T = defaultT, double _Cf = defaultCf );
	~GaussMixDetector() = default;

	void getMotionPicture( const cv::Mat& frame, cv::Mat& motion, bool cleanup = true );

private:
	void Init(const cv::Mat& frame);
	void getpwUpdateAndMotion(cv::Mat& motion);
	void getpwUpdateAndMotionRGB(cv::Mat& motion);
};


#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>

constexpr float defaultT { 0.1f };
constexpr unsigned int defaultHistory { 100U };
constexpr float defaultDeviation { 40.0f };

class GaussMixDetector
{
	// should figure out which parameters can and cannot be changed 'on-the-fly' and
	// add 'set parameter x' function(s) to allow adjusting the model without reinitializing

	static constexpr uchar K { 3U };                           // limit number of Gaussians per pixel
	float alpha { 1 / static_cast<float>(defaultHistory) };    // learning coefficient
	float T { defaultT };                                      // background-foreground threshold
	float initDeviation { defaultDeviation };                  // initial deviation of all Gaussians

	int fRows { 0 }, fCols { 0 }, fChannels { 0 };             // frame parameters

	bool firstFrame { true };                                  // first step flag

	std::vector <cv::Mat> mean {};
	std::vector <cv::Mat> weight {};
	std::vector <cv::Mat> deviation {};
	cv::Mat currentK {};                                        // current number of Gaussians for each pixel

	static const int CVType = cv::DataDepth<float>::value;     // type of 'Mat' pixel info

	std::array<float, 6> mahThreshold                          // list of threshold values for
	{ 3.8416f, 5.9858f, 7.8732f                                // determining ownership probability
	, 6.6564f, 9.245f, 11.6427f };                             // (first row is 95%, second 99%)

public:
	GaussMixDetector() = default;
	explicit GaussMixDetector ( unsigned int _historyLength, double _initDeviation = defaultDeviation, double _T = defaultT);

	void getMotionPicture( const cv::Mat& frame, cv::Mat& motion );

private:
	void Init(const cv::Mat& frame);
	template <typename matPtrType, int channels>
	void getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion);
};


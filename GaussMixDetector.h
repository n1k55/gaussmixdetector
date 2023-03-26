#pragma once

#include <cstdint>
#include <vector>
#include <functional>

#include <opencv2/core/mat.hpp>

constexpr unsigned int defaultHistory { 100U };
constexpr float defaultDeviation { 40.0F };
constexpr float defaultCf { 0.2F };

class GaussMixDetector
{
	// should figure out which parameters can and cannot be changed 'on-the-fly' and
	// add 'set parameter x' function(s) to allow adjusting the model without reinitializing

	static constexpr uchar K { 3U };                           // limit number of Gaussians per pixel
	float alpha { 1 / static_cast<float>(defaultHistory) };    // learning coefficient
	float initDeviation { defaultDeviation };                  // initial deviation of all Gaussians
	float Cf { defaultCf };                                    // model foreground-background threshold

	int fRows { 0 }, fCols { 0 }, fChannels { 0 };             // frame parameters

	bool firstFrame { true };                                  // first step flag

	std::vector <cv::Mat> mean {};
	std::vector <cv::Mat> weight {};
	std::vector <cv::Mat> covariance {};
	cv::Mat currentK {};                                       // current number of Gaussians for each pixel

	bool covTied { true };                                     // flag whether covariances are tied together

	static const int CVType = cv::DataDepth<float>::value;     // type of 'Mat' pixel info
	std::function<void(const cv::Mat&, cv::Mat&)>
		frameProcessor {};

	std::array<float, 6> mahThreshold                          // list of threshold values for
	{ 3.8416F, 5.9858F, 7.8732F                                // determining ownership probability
	, 6.6564F, 9.245F, 11.6427F };                             // (first row is 95%, second 99%)

public:
	GaussMixDetector() = default;
	explicit GaussMixDetector ( unsigned int _historyLength, double _Cf = defaultCf, double _initDeviation = defaultDeviation );

	void getMotionPicture( const cv::Mat& frame, cv::Mat& motion );

private:
	void Init(const cv::Mat& frame);
	void init_processor(const cv::Mat& frame);
	template <typename matPtrType, int channels>
	void getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion);
};


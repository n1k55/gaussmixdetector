#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <opencv2/core/mat.hpp>

constexpr unsigned int defaultHistory { 100U };
constexpr float defaultDeviation { 40.0F };
constexpr float defaultCf { 0.2F };

class GaussMixDetectorD3
{
	// should figure out which parameters can and cannot be changed 'on-the-fly' and
	// add 'set parameter x' function(s) to allow adjusting the model without reinitializing

	static constexpr uchar K { 3U };                           // limit number of Gaussians per pixel
	const float alpha { 1 / static_cast<float>(defaultHistory) };    // learning coefficient
	const float initDeviation { defaultDeviation };                  // initial deviation of all Gaussians
	const float Cf { defaultCf };                                    // model foreground-background threshold

	int fRows { 0 }, fCols { 0 }, fChannels { 0 };             // frame parameters

	bool firstFrame { true };                                  // first step flag

	std::array <cv::Mat, K> mean {};
	std::array <cv::Mat, K> weight {};
	std::array <cv::Mat, K> covariance {};
	cv::Mat currentK {};                                       // current number of Gaussians for each pixel

	// FIXME! No one changes it. Should move to ctor or remove
	static constexpr bool covTied { true };                    // flag whether covariances are tied together

	static constexpr int CVType = cv::DataDepth<float>::value;     // type of 'Mat' pixel info

	static constexpr std::array<float, 6> mahThreshold         // list of threshold values for
	{ 3.8416F, 5.9858F, 7.8732F                                // determining ownership probability
	, 6.6564F, 9.245F, 11.6427F };                             // (first row is 95%, second 99%)

public:
	GaussMixDetectorD3() = default;
	explicit GaussMixDetectorD3( unsigned int _historyLength, double _Cf = defaultCf, double _initDeviation = defaultDeviation );

	void getMotionPicture( const cv::Mat& frame, cv::Mat& motion );

private:
	void Init(const cv::Mat& frame);
	void getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion);
};

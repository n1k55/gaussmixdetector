#include "GaussMixDetectorD3.h"

#include <array>
#include <cassert>
#include <iostream>

#include <opencv2/imgproc.hpp>

namespace {
// Creates a lower triangular identity matrix
cv::Vec<float, 6> symm_eye()
{
	return { 1.F,
			 0.F, 1.F,
			 0.F, 0.F, 1.F
			};
}

// Creates a lower triangular covariance matrix from variances of 'delta'
template <int m>
cv::Vec<float, m * (m + 1) / 2> symm_delta(const cv::Vec<float, m>& delta)
{
	cv::Vec<float, m* (m + 1) / 2> ltm {};

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < i + 1; j++)
		{
			ltm(i * (i + 1) / 2 + j) = delta(i) * delta(j);
		}
	}
	return ltm;
}

float Mahalanobis3(const cv::Vec3f& x, const cv::Vec6f& C)
{
	// Cholesky decomposition
	std::array<float, 6> L {};
	L.at(0) = C(0);
	L.at(1) = C(1) / L.at(0);
	L.at(2) = C(2) - L.at(1) * C(1);
	L.at(3) = C(3) / L.at(0);
	L.at(4) = (C(4) - L.at(3) * C(1)) / L.at(2);
	L.at(5) = C(5) - L.at(3) * C(3) - L.at(4) * L.at(4) * L.at(2);

	// Mahalanobis vector
	std::array<float, 3> y {};

	y.at(0) = x(0) * x(0) / L.at(0);
	y.at(1) = x(1) - x(0) * L.at(1);
	y.at(1) *= y.at(1);
	y.at(1) /= L.at(2);
	y.at(2) = x(2) - L.at(3) * x(0) - L.at(4) * (x(1) - x(0) * L.at(1));
	y.at(2) *= y.at(2);
	y.at(2) /= L.at(5);

	y.at(0) += y.at(1);
	y.at(0) += y.at(2);

	return y.at(0);
}

} // anonymous namespace


GaussMixDetectorD3::GaussMixDetectorD3( unsigned int _historyLength, double _Cf, double _initDeviation)
	: alpha { 1 / static_cast<float>(_historyLength) }
	, initDeviation { static_cast<float>(_initDeviation) }
	, Cf { static_cast<float>(_Cf) }
{
}

void GaussMixDetectorD3::Init( const cv::Mat& frame )
{
	assert(!frame.empty());
	if (frame.rows < 1 || frame.cols < 1)
	{
		throw std::invalid_argument("Image has invalid size.");
	}

	fRows = frame.rows;
	fCols = frame.cols;
	fChannels = frame.channels();

	mean.fill({fRows, fCols, CV_MAKETYPE(CVType, fChannels), cv::Scalar(0, 0)});
	// Initialise the first Gaussian's mean with the first frame
	cv::Mat tmp;
	frame.convertTo( tmp, CV_MAKETYPE(CVType, fChannels ) );
	mean.front() = tmp;

	weight.fill({fRows, fCols, CV_MAKETYPE(CVType, 1), cv::Scalar(0)});
	// Initialise the first Gaussian's weight with alpha
	weight.front() = cv::Mat(fRows, fCols, CV_MAKETYPE(CVType, 1 ), cv::Scalar( alpha ));

	// The magic below comes down to the task of storing the
	// symmetrical covariance matrix as lower triangular matrix
	// for efficiency.
	// We write and read it from top to bottom, left to right:
	// (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), ...

	const int covChannels = fChannels * (fChannels + 1) / 2;
	cv::Mat pattern(1, covChannels, CVType, cv::Scalar(0));

	// Initalise the diagonal of all first Gaussian's
	// cov. matrixes with initial deviation parameter
	auto* p = pattern.ptr<float>(0);
	for (int c = 1; c < fChannels+1; c++)
	{
		p[c * (c + 1) / 2 - 1] = initDeviation;
	}
	const auto covariance_initial { cv::repeat(pattern, fRows, fCols).reshape(covChannels) };
	// populate the cov. vector with pointers to first cov. image
	if (covTied)
	{
		covariance.fill(covariance_initial);
	}
	// otherwise init with zeros as with other parameters
	else
	{
		covariance.fill({fRows, fCols, CV_MAKETYPE(CVType, covChannels), cv::Scalar(0)});
		weight.front() = covariance_initial;
	}

	// Current number of Gaussians is 1 for all pixels
	currentK = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 )
			, cv::Scalar( 1 ) );

	firstFrame = false;
}

void GaussMixDetectorD3::getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion)
{
	// template parameters, instatiated
	constexpr int channels {3};
	using matPtrType = uchar;

	constexpr int covChannels = channels * (channels + 1) / 2;

	if (sizeof(matPtrType) != frame.elemSize1())
	{
		// TODO: message wording and error details
		std::cerr << "Hello !" << sizeof(matPtrType) <<' ' << frame.elemSize1() << std::endl;
		throw std::runtime_error("Environment specific error. cv::Mat element \
			size mismatch.");
	}

	// mb use cv::Mat::forEach
	for( int i = 0; i < fRows; i++ )
	{
		const auto* framePtr = frame.ptr<matPtrType>(i);
		auto* motionPtr = motion.ptr<uchar>(i);
		auto* currentKPtr = currentK.ptr<uchar>(i);
		// Vec structures hold information accross channels
		// Mean value of each Gaussian
		std::array<cv::Vec<float, channels>*, K> meanPtr {};
		// Lower triangular of Covariance matrix of each Gaussian
		std::array<cv::Vec<float, covChannels>*, K> covariancePtr {};
		// Weight of each Gaussians
		std::array<float*, K> weightPtr {};
		for ( uchar k = 0U; k < K; k++ )
		{
			meanPtr.at(k) = mean[k].ptr<cv::Vec<float, channels>>(i);
			covariancePtr.at(k) = covariance[k].ptr<cv::Vec<float, covChannels>>(i);
			weightPtr.at(k) = weight[k].ptr<float>(i);
		}

		for ( int j = 0; j < fCols; j++ )
		{
			const int iRGB = j*channels;

			// Pixel value from input image (cast to floating point)
			// pixelVal holds (B, G, R) values in case
			// input image is of standard 3-channel RGB type
			cv::Vec<float, channels> pixelVal;
			for (int c = 0; c < channels; c++)
			{
				pixelVal(c) = static_cast<float>(framePtr[iRGB + c]);
			}
			uchar currentPixelK = currentKPtr[j];
			// The distance (difference) between mean and target vector (pixel)
			std::array<cv::Vec<float, channels>, K> delta {};
			for ( uchar k = 0U; k < currentPixelK; k++ )
			{
				delta.at(k) = pixelVal - meanPtr.at(k)[j];
			}

			// Whether current pixel 'belongs' to k-th Gaussian
			std::array<bool, K> isCurrent {};

			uchar owner = 0U;
			for ( owner = 0U; owner < currentPixelK; owner++ )
			{
				if ( Mahalanobis3(delta.at(owner), covariancePtr.at(owner)[j]) < GaussMixDetectorD3::mahThreshold.at(channels - 1 + 3) )
				{
					isCurrent.at(owner) = true;
					break;
				}
			}

			if( owner == currentPixelK )
			{
				if ( currentPixelK < K )
				{
					meanPtr.at(currentPixelK)[j] = pixelVal;
					weightPtr.at(currentPixelK)[j] = alpha;
					if (!covTied)
					{
						covariancePtr.at(currentPixelK)[j] = initDeviation * symm_eye();
					}
					currentPixelK++;
				}
				else
				{
					meanPtr.at(K-1U)[j] = pixelVal;
					weightPtr.at(K-1U)[j] = alpha;
					if (!covTied)
					{
						covariancePtr.at(K-1U)[j] = initDeviation * symm_eye();
					}
				}
			}
			else
			{
				const float w = (alpha / weightPtr.at(owner)[j]);
				meanPtr.at(owner)[j] += w * delta.at(owner);
				float tied_multiplier = 1.F;
				if (covTied)
				{
					tied_multiplier /= static_cast<float>(currentPixelK);
				}
				covariancePtr.at(owner)[j] += std::min( 20*alpha, w ) * (symm_delta(delta.at(owner)) - covariancePtr.at(owner)[j]) * tied_multiplier;
				covariancePtr.at(owner)[j] += 10 * alpha * symm_eye();
			}

			{
				float w = 0;
				for ( uchar k = 0U; k < currentPixelK; k++ )
				{
					weightPtr.at(k)[j] = isCurrent.at(k) ? weightPtr.at(k)[j] * (1 - alpha) + alpha : weightPtr.at(k)[j] * (1 - alpha);
					w += weightPtr.at(k)[j];
				}

				for ( uchar k = 0U; k < currentPixelK; k++ )
				{
					weightPtr.at(k)[j] = weightPtr.at(k)[j] / w;
				}
			}

			bool noMov = false;
			while (!noMov)
			{
				noMov = true;
				for ( uchar k = 0U; k < currentPixelK-1U; k++ )
				{
					if ( weightPtr.at(k)[j] < weightPtr.at(k+1U)[j] )
					{
						std::swap(weightPtr.at(k)[j], weightPtr.at(k+1U)[j]);
						cv::swap(meanPtr.at(k)[j], meanPtr.at(k+1U)[j]);
						cv::swap(covariancePtr.at(k)[j], covariancePtr.at(k+1U)[j]);
						std::swap(isCurrent.at(k), isCurrent.at(k+1U));

						noMov = false;
					}
				}
			}

			for (owner = 0U; owner < currentPixelK; owner++)
			{
				if (isCurrent.at(owner))
				{
					break;
				}
			}

			float w = weightPtr.at(0)[j];
			for (uchar bgCount = 1U; bgCount < currentPixelK; bgCount++)
			{
				if (w <= (1.F - Cf))
				{
					w += weightPtr.at(bgCount)[j];
				}
				else
				{
					if (owner < bgCount)
					{
						motionPtr[j] = 0U;
					}
					else
					{
						motionPtr[j] = 255U;
					}
					break;
				}
			}

			currentKPtr[j] = currentPixelK;
		}
	}
}

void GaussMixDetectorD3::getMotionPicture( const cv::Mat& frame, cv::Mat& motion )
{
	if (frame.empty())
	{
		throw std::invalid_argument("No input image.");
	}

	if (firstFrame)
	{
		Init(frame);
		motion = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 ), cv::Scalar( 0 ) );
		return;
	}

	if (fRows != frame.rows || fCols != frame.cols)
	{
		throw std::invalid_argument("Input image size different from initial. Stream must be uniform.");
	}
	if (fChannels != frame.channels())
	{
		throw std::invalid_argument("Input image channels different from initial. Stream must be uniform.");
	}

	motion = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 ), cv::Scalar( 0 ) );

	getpwUpdateAndMotionRGB(frame, motion);
}

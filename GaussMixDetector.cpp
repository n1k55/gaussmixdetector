#include "GaussMixDetector.h"

#include <array>
#include <cassert>

#include <opencv2/imgproc.hpp>


GaussMixDetector::GaussMixDetector( unsigned int _historyLength, double _initDeviation, double _T, double _Cf )
	: alpha { 1 / static_cast<double>(_historyLength) }
	, T { _T }
	, initDeviation { _initDeviation }
	, Cf { _Cf }
{
}

void GaussMixDetector::Init( const cv::Mat& frame )
{
	assert(!frame.empty());
	if (frame.rows < 1 || frame.cols < 1)
	{
		throw std::invalid_argument("Image has invalid size.");
	}

	cv::Mat tmp;
	fRows = frame.rows;
	fCols = frame.cols;
	fChannels = frame.channels();

	// Initialise the first Gaussian's mean with the first frame
	frame.convertTo( tmp, CV_MAKETYPE( CVType, fChannels ) );
	mean.push_back( tmp );

	// Initialise the first Gaussian's weight with alpha
	weight.emplace_back(fRows, fCols, CV_MAKETYPE( CVType, 1 ), cv::Scalar( alpha ));

	// The magic below comes down to the task of storing the
	// symmetrical covariance matrix as lower triangular matrix
	// for efficiency.
	// We write and read it from top to bottom, left to right:
	// (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), ...

	const int devChannels = fChannels * (fChannels + 1) / 2;
	cv::Mat pattern(1, devChannels, CVType, cv::Scalar(0));

	// Initalise the diagonal of all first Gaussian's
	// cov. matrixes with initial deviation parameter
	auto* p = pattern.ptr<double>(0);
	for (int c = 1; c < fChannels+1; c++)
	{
		p[c * (c + 1) / 2 - 1] = initDeviation;
	}
	deviation.push_back(cv::repeat(pattern, fRows, fCols).reshape(devChannels));

	// Initialise the rest of parameters with zeros
	for (int k = 1; k < K; k++)
	{
		mean.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, fChannels), cv::Scalar(0, 0));
		weight.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, 1), cv::Scalar(0));
		deviation.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, devChannels), cv::Scalar(0));
	}

	// Current number of Gaussians is 1 for all pixels
	currentK = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 )
			, cv::Scalar( 1 ) );

	firstFrame = false;
}

// Extracts a lower triangular matrix from a square matrix
template <int m>
cv::Matx<double, 1, m*(m+1)/2> symm_extract(const cv::Matx<double, m, m>& matrix)
{
	cv::Matx<double, 1, m * (m + 1) / 2> ltm {};
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < i + 1; j++)
		{
			ltm(i * (i + 1) / 2 + j) = matrix(i, j);
		}
	}

	return ltm;
}

// Creates a lower triangular identity matrix
template <int m>
cv::Matx<double, 1, m * (m + 1) / 2> symm_eye()
{
	cv::Matx<double, 1, m * (m + 1) / 2> ltm {};

	for (int c = 1; c < m + 1; c++)
	{
		ltm(c * (c + 1) / 2 - 1) = 1.0;
	}
	return ltm;
}

// Creates a lower triangular covariance matrix from variances of 'delta'
template <int m>
cv::Matx<double, 1, m* (m + 1) / 2> symm_delta(const cv::Matx<double, 1, m>& delta)
{
	cv::Matx<double, 1, m* (m + 1) / 2> ltm {};

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < i + 1; j++)
		{
			ltm(i * (i + 1) / 2 + j) = delta(i) * delta(j);
		}
	}
	return ltm;
}

template <int channels>
double Mahalanobis(const cv::Matx<double, 1, channels>& x, const cv::Matx<double, 1, channels*(channels+1)/2>& C)
{
	cv::Matx<double, channels, channels> cov {};
	for (int i = 0; i < channels; i++)
	{
		for (int j = 0; j < channels; j++)
		{
			cov(i, j) = cov(j, i) = C(i*(i+1)/2 + j);
		}
	}
	return (x * cov.inv(cv::DECOMP_CHOLESKY)).dot(x);
}

template <>
double Mahalanobis<1>(const cv::Matx<double, 1, 1>& x, const cv::Matx<double, 1, 1>& C)
{
	return x(0) * x(0) / C(0);
}

template <>
double Mahalanobis<2>(const cv::Matx12d& x, const cv::Matx13d& C)
{
	// Cholesky decomposition
	std::array<double, 3> L {};
	L.at(0) = C(0);
	L.at(1) = C(1) / L.at(0);
	L.at(2) = C(2) - L.at(1) * C(1);

	// Mahalanobis vector
	double y = x(1) - x(0) * L.at(1);
	y *= y;
	y /= L.at(2);
	y += x(0) * x(0) / L.at(0);

	return y;
}

template <>
double Mahalanobis<3>(const cv::Matx13d& x, const cv::Matx16d& C)
{
	// Cholesky decomposition
	std::array<double, 6> L {};
	L.at(0) = C(0);
	L.at(1) = C(1) / L.at(0);
	L.at(2) = C(2) - L.at(1) * C(1);
	L.at(3) = C(3) / L.at(0);
	L.at(4) = (C(4) - L.at(3) * C(1)) / L.at(2);
	L.at(5) = C(5) - L.at(3) * C(3) - L.at(4) * L.at(4) * L.at(2);

	// Mahalanobis vector
	std::array<double, 3> y {};

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

template <typename matPtrType, int channels>
void GaussMixDetector::getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion)
{
	// Matx structures hold information accross channels,
	// e.g. pixelVal holds (B, G, R) values in case
	// input image is of standard 3-channel RGB type

	// Pixel value from input image (cast to floating point)
	cv::Matx<double, 1, channels> pixelVal;
	// Mean value of each Gaussian
	std::array<cv::Matx<double, 1, channels>*, K> meanVal {};
	
	const int devChannels = channels * (channels + 1) / 2;
	// Lower triangular of Covariance matrix of each Gaussian
	std::array<cv::Matx<double, 1, devChannels>*, K> deviationVal {};

	// Weight of each Gaussians
	std::array<double*, K> weightVal {};

	// The distance (difference) between mean and target vector (pixel)
	std::array<cv::Matx<double, 1, channels>, K> delta {};
	// Whether current pixel 'belongs' to k-th Gaussian
	std::array<bool, K> isCurrent {};

	for( int i = 0; i < fRows; i++ )
	{
		const auto* framePtr = frame.ptr<matPtrType>(i);
		auto* motionPtr = motion.ptr<uchar>(i);
		auto* currentKPtr = currentK.ptr<uchar>(i);
		for ( uchar k = 0U; k < K; k++ )
		{
			meanVal.at(k) = mean[k].ptr<cv::Matx<double, 1, channels>>(i);
			deviationVal.at(k) = deviation[k].ptr<cv::Matx<double, 1, devChannels>>(i);
			weightVal.at(k) = weight[k].ptr<double>(i);
		}

		uchar currentPixelK = 0U;
		for ( int j = 0; j < fCols; j++ )
		{
			const int iRGB = j*channels;

			for (int c = 0; c < channels; c++)
			{
				pixelVal(c) = static_cast<double>(framePtr[iRGB + c]);
			}
			currentPixelK = currentKPtr[j];
			for ( uchar k = 0U; k < currentPixelK; k++ )
			{
				delta.at(k) = pixelVal - meanVal.at(k)[j];
			}

			uchar count = 0U;
			for ( uchar k = 0U; k < currentPixelK; k++ )
			{
				// this doesn't seem right - need to find a suitable threshold for 'belonging' condition
				if ( Mahalanobis(delta.at(k), deviationVal.at(k)[j]) < sqrt(cv::trace(deviationVal.at(k)[j])) )
				{
					count++;
					isCurrent.at(k) = true;
				}
				else
				{
					isCurrent.at(k) = false;
				}
			}

			if( count == 0U )
			{
				if ( currentPixelK < K )
				{
					meanVal.at(currentPixelK)[j] = pixelVal;
					deviationVal.at(currentPixelK)[j] = initDeviation * symm_eye<channels>();
					weightVal.at(currentPixelK)[j] = alpha;
					currentPixelK++;
				}
				else
				{
					meanVal.at(K-1U)[j] = pixelVal;
					deviationVal.at(K-1U)[j] = initDeviation * symm_eye<channels>();
					weightVal.at(K-1U)[j] = alpha;
				}
			}
			else
			{
				for ( uchar k = 0U; k < currentPixelK; k++ )
				{
					if ( isCurrent.at(k) )
					{
						const double w = (alpha / weightVal.at(k)[j]);
						meanVal.at(k)[j] += w * delta.at(k);
						deviationVal.at(k)[j] += std::min( 20*alpha, w ) * symm_delta(delta.at(k));
					}
				}
			}

			{
				double w = 0;
				for ( uchar k = 0U; k < currentPixelK; k++ )
				{
					weightVal.at(k)[j] = weightVal.at(k)[j] * (1-alpha) + alpha*int(isCurrent.at(k));
					w += weightVal.at(k)[j];
				}

				for ( uchar k = 0U; k < currentPixelK; k++ )
				{
					weightVal.at(k)[j] = weightVal.at(k)[j] / w;
				}
			}

			bool noMov = false;
			while (!noMov)
			{
				noMov = true;
				for ( uchar k = 0U; k < currentPixelK-1U; k++ )
				{
					if ( weightVal.at(k)[j] < weightVal.at(k+1U)[j] )
					{
						std::swap(weightVal.at(k)[j], weightVal.at(k+1U)[j]);
						cv::swap(meanVal.at(k)[j], meanVal.at(k+1U)[j]);
						cv::swap(deviationVal.at(k)[j], deviationVal.at(k+1U)[j]);

						noMov = false;
					}
				}
			}

			bool FG = sqrt(delta[0].dot(delta[0])) > T;

			if( FG )
			{
				motionPtr[j] = 255U;
			}

			currentKPtr[j] = currentPixelK;
		}
	}
}

void GaussMixDetector::getMotionPicture( const cv::Mat& frame, cv::Mat& motion )
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

	const std::invalid_argument depth_exception("Unknown cv::Mat depth. Accepted depths: \
		CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.");
	switch (fChannels)
	{
		case 1:
			switch (frame.depth())
			{
				case CV_8U:
					getpwUpdateAndMotionRGB<uchar, 1>(frame, motion);
					break;
				case CV_8S:
					getpwUpdateAndMotionRGB<schar, 1>(frame, motion);
					break;
				case CV_16U:
					getpwUpdateAndMotionRGB<ushort, 1>(frame, motion);
					break;
				case CV_16S:
					getpwUpdateAndMotionRGB<short, 1>(frame, motion);
					break;
				case CV_32S:
					getpwUpdateAndMotionRGB<int, 1>(frame, motion);
					break;
				case CV_32F:
					getpwUpdateAndMotionRGB<float, 1>(frame, motion);
					break;
				case CV_64F:
					getpwUpdateAndMotionRGB<double, 1>(frame, motion);
					break;
				default:
					throw depth_exception;
			}
			break;
		case 2:
			switch (frame.depth())
			{
				case CV_8U:
					getpwUpdateAndMotionRGB<uchar, 2>(frame, motion);
					break;
				case CV_8S:
					getpwUpdateAndMotionRGB<schar, 2>(frame, motion);
					break;
				case CV_16U:
					getpwUpdateAndMotionRGB<ushort, 2>(frame, motion);
					break;
				case CV_16S:
					getpwUpdateAndMotionRGB<short, 2>(frame, motion);
					break;
				case CV_32S:
					getpwUpdateAndMotionRGB<int, 2>(frame, motion);
					break;
				case CV_32F:
					getpwUpdateAndMotionRGB<float, 2>(frame, motion);
					break;
				case CV_64F:
					getpwUpdateAndMotionRGB<double, 2>(frame, motion);
					break;
				default:
					throw depth_exception;
			}
			break;
		case 3:
			switch (frame.depth())
			{
				case CV_8U:
					getpwUpdateAndMotionRGB<uchar, 3>(frame, motion);
					break;
				case CV_8S:
					getpwUpdateAndMotionRGB<schar, 3>(frame, motion);
					break;
				case CV_16U:
					getpwUpdateAndMotionRGB<ushort, 3>(frame, motion);
					break;
				case CV_16S:
					getpwUpdateAndMotionRGB<short, 3>(frame, motion);
					break;
				case CV_32S:
					getpwUpdateAndMotionRGB<int, 3>(frame, motion);
					break;
				case CV_32F:
					getpwUpdateAndMotionRGB<float, 3>(frame, motion);
					break;
				case CV_64F:
					getpwUpdateAndMotionRGB<double, 3>(frame, motion);
					break;
				default:
					throw depth_exception;
			}
			break;
		default:
			throw std::invalid_argument("Accepted number of channels: 1 through 3.");
	}
}

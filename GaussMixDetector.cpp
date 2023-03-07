#include "GaussMixDetector.h"

#include <array>
#include <cassert>

#include <opencv2/imgproc.hpp>


GaussMixDetector::GaussMixDetector( unsigned int _historyLength, double _initDeviation, double _T)
	: alpha { 1 / static_cast<float>(_historyLength) }
	, T { static_cast<float>(_T) }
	, initDeviation { static_cast<float>(_initDeviation) }
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
	frame.convertTo( tmp, CV_MAKETYPE(CVType, fChannels ) );
	mean.push_back( tmp );

	// Initialise the first Gaussian's weight with alpha
	weight.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, 1 ), cv::Scalar( alpha ));

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
	covariance.push_back(cv::repeat(pattern, fRows, fCols).reshape(covChannels));

	// Initialise the rest of parameters with zeros
	for (int k = 1; k < K; k++)
	{
		mean.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, fChannels), cv::Scalar(0, 0));
		weight.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, 1), cv::Scalar(0));
		covariance.emplace_back(fRows, fCols, CV_MAKETYPE(CVType, covChannels), cv::Scalar(0));
	}

	// Current number of Gaussians is 1 for all pixels
	currentK = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 )
			, cv::Scalar( 1 ) );

	firstFrame = false;
}

// Extracts a lower triangular matrix from a square matrix
template <int m>
cv::Vec<float, m*(m+1)/2> symm_extract(const cv::Matx<float, m, m>& matrix)
{
	cv::Vec<float, m * (m + 1) / 2> ltm {};
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
cv::Vec<float, m * (m + 1) / 2> symm_eye()
{
	cv::Vec<float, m * (m + 1) / 2> ltm {};

	for (int c = 1; c < m + 1; c++)
	{
		ltm(c * (c + 1) / 2 - 1) = 1.0;
	}
	return ltm;
}

// Creates a lower triangular covariance matrix from variances of 'delta'
template <int m>
cv::Vec<float, m* (m + 1) / 2> symm_delta(const cv::Vec<float, m>& delta)
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

template <int channels>
float Mahalanobis(const cv::Vec<float, channels>& x, const cv::Vec<float, channels*(channels+1)/2>& C)
{
	cv::Matx<float, channels, channels> cov {};
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
float Mahalanobis<1>(const cv::Vec<float, 1>& x, const cv::Vec<float, 1>& C)
{
	return x(0) * x(0) / C(0);
}

template <>
float Mahalanobis<2>(const cv::Vec2f& x, const cv::Vec3f& C)
{
	// Cholesky decomposition
	std::array<float, 3> L {};
	L.at(0) = C(0);
	L.at(1) = C(1) / L.at(0);
	L.at(2) = C(2) - L.at(1) * C(1);

	// Mahalanobis vector
	float y = x(1) - x(0) * L.at(1);
	y *= y;
	y /= L.at(2);
	y += x(0) * x(0) / L.at(0);

	return y;
}

template <>
float Mahalanobis<3>(const cv::Vec3f& x, const cv::Vec6f& C)
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

template <typename matPtrType, int channels>
void GaussMixDetector::getpwUpdateAndMotionRGB(const cv::Mat& frame, cv::Mat& motion)
{
	if (sizeof(matPtrType) != frame.elemSize1())
	{
		// TODO: message wording and error details
		throw std::runtime_error("Environment specific error. cv::Mat element \
			size mismatch.");
	}

	// Vec structures hold information accross channels,
	// e.g. pixelVal holds (B, G, R) values in case
	// input image is of standard 3-channel RGB type

	// Pixel value from input image (cast to floating point)
	cv::Vec<float, channels> pixelVal;
	// Mean value of each Gaussian
	std::array<cv::Vec<float, channels>*, K> meanPtr {};

	const int covChannels = channels * (channels + 1) / 2;
	// Lower triangular of Covariance matrix of each Gaussian
	std::array<cv::Vec<float, covChannels>*, K> covariancePtr {};

	// Weight of each Gaussians
	std::array<float*, K> weightPtr {};

	// The distance (difference) between mean and target vector (pixel)
	std::array<cv::Vec<float, channels>, K> delta {};

	for( int i = 0; i < fRows; i++ )
	{
		const auto* framePtr = frame.ptr<matPtrType>(i);
		auto* motionPtr = motion.ptr<uchar>(i);
		auto* currentKPtr = currentK.ptr<uchar>(i);
		for ( uchar k = 0U; k < K; k++ )
		{
			meanPtr.at(k) = mean[k].ptr<cv::Vec<float, channels>>(i);
			covariancePtr.at(k) = covariance[k].ptr<cv::Vec<float, covChannels>>(i);
			weightPtr.at(k) = weight[k].ptr<float>(i);
		}

		uchar currentPixelK = 0U;
		for ( int j = 0; j < fCols; j++ )
		{
			const int iRGB = j*channels;

			for (int c = 0; c < channels; c++)
			{
				pixelVal(c) = static_cast<float>(framePtr[iRGB + c]);
			}
			currentPixelK = currentKPtr[j];
			for ( uchar k = 0U; k < currentPixelK; k++ )
			{
				delta.at(k) = pixelVal - meanPtr.at(k)[j];
			}

			// Whether current pixel 'belongs' to k-th Gaussian
			std::array<bool, K> isCurrent {};

			uchar owner = 0U;
			for ( owner = 0U; owner < currentPixelK; owner++ )
			{
				if ( Mahalanobis(delta.at(owner), covariancePtr.at(owner)[j]) < mahThreshold.at(channels - 1 + 3) )
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
					covariancePtr.at(currentPixelK)[j] = initDeviation * symm_eye<channels>();
					weightPtr.at(currentPixelK)[j] = alpha;
					currentPixelK++;
				}
				else
				{
					meanPtr.at(K-1U)[j] = pixelVal;
					covariancePtr.at(K-1U)[j] = initDeviation * symm_eye<channels>();
					weightPtr.at(K-1U)[j] = alpha;
				}
			}
			else
			{
				const float w = (alpha / weightPtr.at(owner)[j]);
				meanPtr.at(owner)[j] += w * delta.at(owner);
				covariancePtr.at(owner)[j] += std::min( 20*alpha, w ) * (symm_delta(delta.at(owner)) - covariancePtr.at(owner)[j]);
				covariancePtr.at(owner)[j] += 10 * alpha * symm_eye<channels>();
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

	auto make_depth_exception = [] {
		return std::invalid_argument("Unknown cv::Mat depth. Accepted depths: \
		CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.");
	};

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
					throw make_depth_exception();
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
					throw make_depth_exception();
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
					throw make_depth_exception();
			}
			break;
		default:
			throw std::invalid_argument("Accepted number of channels: 1 through 3.");
	}
}

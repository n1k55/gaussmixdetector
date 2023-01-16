#include "GaussMixDetector.h"

#include <opencv2/imgproc.hpp>

const static cv::Mat smCircle = (cv::Mat_<uchar>(5, 5) << 0, 0, 1, 0, 0,
														  0, 1, 1, 1, 0,
														  1, 1, 1, 1, 1,
														  0, 1, 1, 1, 0,
														  0, 0, 1, 0, 0);

GaussMixDetector::GaussMixDetector()
{
	historyLength = defaultHistory;
	initDeviation = defaultDeviation;
	alpha = double(1) / double(historyLength);
	T = defaultT;
	Cf = defaultCf;

	firstFrame = true;

	fRows = 0;
	fCols = 0;
	fChannels = 0;
}

GaussMixDetector::GaussMixDetector( unsigned int _historyLength, double _initDeviation, double _T, double _Cf )
{
	historyLength = _historyLength;
	initDeviation = _initDeviation;
	alpha = double(1) / double(_historyLength);
	T = _T;
	Cf = _Cf;

	firstFrame = true;

	fRows = 0;
	fCols = 0;
	fChannels = 0;
}

void GaussMixDetector::Init( const cv::Mat& frame )
{
	assert(!frame.empty());
	if (frame.rows < 1 || frame.cols < 1)
	{
		throw std::invalid_argument("Image has invalid size.");
	}

	cv::Mat tmp;
	ptrType* p;
	fRows = frame.rows;
	fCols = frame.cols;
	fChannels = frame.channels();

	frame.convertTo( tmp, CV_MAKETYPE( CVType, fChannels ) );

	mean.push_back( tmp );
	weight.push_back( cv::Mat( fRows, fCols, CV_MAKETYPE( CVType, 1 )
			, cv::Scalar( 1 ) ) );

	for ( int k = 1; k < K; k++ )
	{
		mean.push_back( cv::Mat( fRows, fCols, CV_MAKETYPE( CVType, fChannels )
			, cv::Scalar( 0, 0 ) ) );
		weight.push_back( cv::Mat( fRows, fCols, CV_MAKETYPE( CVType, 1 )
			, cv::Scalar( 0 ) ) );
	}

	tmp.create( fRows, fCols, CV_MAKETYPE( CVType, fChannels*fChannels ) );
	for ( int k = 0; k < K; k++ )
	{
		for( int r = 0; r < fRows; r++ )
		{
			p = tmp.ptr<double>(r);
			for( int c = 0; c < fCols*fChannels*fChannels; c++ )
			{
				p[c] = ( (c % (fChannels*fChannels)) % (fChannels+1) == 0 ) ? initDeviation : 0;
			}
		}
		deviation.push_back( tmp );
	}

	currentK = cv::Mat( fRows, fCols, CV_MAKETYPE( CV_8U, 1 )
			, cv::Scalar( 1 ) );

	firstFrame = false;
}

inline double normDistrib( int x, double m, double d )
{
	double tmp = (double(x) - m) / d;
	tmp *= tmp;
	tmp = exp( -tmp/2 );
	return tmp / d / sqrt(2*CV_PI);
}

inline double normDistrib( double x, double m, double d )
{
	double tmp = (x - m) / d;
	tmp *= tmp;
	tmp = exp( -tmp/2 );
	return tmp / d / sqrt(2*CV_PI);
}

void GaussMixDetector::getpwUpdateAndMotion( cv::Mat& motion )
{
	int i,j,motionI;
	short k,count;
	double w, r;

	const ptrType* ptF;
	uchar* ptMo;
	uchar* ptK;

	std::vector<ptrType*> ptM(K);
	std::vector<ptrType*> ptD(K);
	std::vector<ptrType*> ptW(K);
	std::vector<bool> isCurrent(K,false);

	double tmpF;
	short tmpK;

	double tmpM[K] {};
	double tmpD[K] {};
	double tmpW[K] {};

	for( i = 0; i < fRows; i++ )
	{
		ptF = fClone.ptr<double>(i);
		ptMo = motion.ptr<uchar>(i);
		ptK = currentK.ptr<uchar>(i);
		for ( k = 0; k < K; k++ )
		{
			ptM[k] = mean[k].ptr<double>(i);
			ptD[k] = deviation[k].ptr<double>(i);
			ptW[k] = weight[k].ptr<double>(i);
		}
		for ( j = 0; j < fCols*fChannels; j += fChannels )
		{
			count = 0;
			w = 0;
			tmpF = ptF[j];
			tmpK = ptK[j];
			for ( k = 0; k < tmpK; k++ )
			{
				tmpM[k] = ptM[k][j];
				tmpD[k] = ptD[k][j];
				tmpW[k] = ptW[k][j];
			}

			for ( k = 0; k < tmpK; k++ )
			{
				isCurrent[k] = false;
				if (( tmpF < tmpM[k] + tmpD[k] ) && ( tmpF > tmpM[k] - tmpD[k] ))
				{
					count++;
					isCurrent[k] = true;
				}
				else
				{
					isCurrent[k] = false;
				}
			}
			if( count == 0 )
			{
				if ( tmpK < K )
				{
					tmpM[tmpK] = tmpF;
					tmpD[tmpK] = initDeviation;
					tmpW[tmpK] = 0;
					isCurrent[tmpK] = true;
					tmpK = tmpK + 1;
				}
				else
				{
					tmpM[0] = tmpF;
					tmpD[0] = initDeviation;
					isCurrent[0] = true;
					tmpW[0] = 0;
				}
			}
			else
			{
				for ( k = 0; k < tmpK; k++ )
				{
					if ( isCurrent[k] ) {
						r = alpha * normDistrib( tmpF, tmpM[k], tmpD[k] );
						tmpD[k] = sqrt ( (1-r) / 6.25 * tmpD[k] * tmpD[k] + r * (tmpF - tmpM[k]) );
						tmpM[k] = (1-r) * tmpM[k] + r*tmpF;
					}

					if ( isCurrent[k] ) {
						tmpM[k] = (1-alpha) * tmpM[k] + alpha * tmpF;
						tmpD[k] = sqrt ( (1-alpha) * tmpD[k] * tmpD[k] / 6.25
							+ alpha * (tmpF - tmpM[k]) * (tmpF - tmpM[k]) );
					}
				}
			}
			for ( k = 0; k < tmpK; k++ )
			{
				tmpW[k] = tmpW[k] * (1-alpha) + alpha*int(isCurrent[k]);
				w += tmpW[k];
			}

			for ( k = 0; k < tmpK; k++ )
			{
				tmpW[k] = tmpW[k] / w;
			}

			w = 0;
			for ( k = 0; k < tmpK; k++ )
			{
				w += tmpW[k];
			}

			assert( w == 1.0 );

			bool noMov = false;
			while (!noMov)
			{
				noMov = true;
				for ( k = 0; k < tmpK-1; k++ )
				{
					if ( tmpW[k] > tmpW[k+1] )
					{
						w = tmpW[k];
						tmpW[k] = tmpW[k+1];
						tmpW[k+1] = w;
						w = tmpM[k];
						tmpM[k] = tmpM[k+1];
						tmpM[k+1] = w;
						w = tmpD[k];
						tmpD[k] = tmpD[k+1];
						tmpD[k+1] = w;
						noMov = false;
					}
				}
			}

			motionI = j / fChannels;

			count = -1;
			w = tmpW[0];

			for ( k = 1; k < tmpK; k++ )
			{
				if ( w >= Cf ) {
					count = k;
					break;
				}

				w += tmpW[k];
			}
			if ( count != -1 )
			{
				for ( k = 0; k < count; k++ )
				{
					if(isCurrent[k])
					{
						ptMo[motionI] = 255;
					}
				}
			}

			if( tmpK > 1 )
			{
				bool isMotion = true;
				for ( k = 1; k < tmpK; k++ )
				{
					if( !(abs(tmpM[0] - tmpM[k]) > T) )
					{
						isMotion = false;
						break;
					}
				}

				if( isMotion )
				{
					ptMo[motionI] = 255;
				}
			}

			ptK[j] = static_cast<uchar>(tmpK);
			for ( k = 0; k < tmpK; k++ )
			{
				ptM[k][j] = tmpM[k];
				ptD[k][j] = tmpD[k];
				ptW[k][j] = tmpW[k];
			}
		}
	}
}

double toDouble( cv::MatExpr M )
{
	cv::Mat A(M);
	assert(A.rows == 1 || A.cols == 1);
	return A.at<double>(0);
}

inline double Mahalanobis(cv::Matx13d delta, cv::Matx33d C)
{
	C = C.inv();
	return (delta * C).dot(delta);
}

inline double Mahalanobis( cv::Matx13d x, cv::Matx13d m, cv::Matx33d C )
{
	return Mahalanobis( (x-m), C );
}

inline double normDistrib3( cv::Matx13d x, cv::Matx13d m, cv::Matx33d C )
{
	double mah = Mahalanobis( x, m, C );
	return exp( - mah / 2 ) / 2 / CV_PI / sqrt(2*CV_PI); // det;
}

void GaussMixDetector::getpwUpdateAndMotionRGB( cv::Mat& motion )
{
	int i,j,c,cd,iRGB,iDev;
	short k,count;
	double w;

	const ptrType* ptF;
	uchar* ptMo;
	uchar* ptK;

	ptrType* ptM[K];
	ptrType* ptD[K];
	ptrType* ptW[K];
	bool isCurrent[K] {};
	int tmpK;

	cv::Matx13d tmpF;
	cv::Matx13d delta[K];
	cv::Matx13d tmpM[K];
	cv::Matx33d tmpD[K];
	cv::Matx33d dhelp;
	double tmpW[K] {};

	for( i = 0; i < fRows; i++ )
	{
		ptF = fClone.ptr<double>(i);
		ptMo = motion.ptr<uchar>(i);
		ptK = currentK.ptr<uchar>(i);
		for ( k = 0; k < K; k++ )
		{
			ptM[k] = mean[k].ptr<double>(i);
			ptD[k] = deviation[k].ptr<double>(i);
			ptW[k] = weight[k].ptr<double>(i);
		}
		for ( j = 0; j < fCols; j++ )
		{
			count = 0;
			w = 0;
			iRGB = j*fChannels;
			iDev = j*fChannels*fChannels;
			// !!! why 3 ??
			for ( c = 0; c < 3; c++ )
			{
				tmpF(c) = ptF[iRGB + c];
			}
			tmpK = ptK[j];

			for ( k = 0; k < tmpK; k++ )
			{
				for ( c = 0; c < 3; c++ )
				{
					tmpM[k](c) = ptM[k][iRGB + c];
					for ( cd = 0; cd < 3; cd++ )
					{
						tmpD[k](c,cd) = ptD[k][iDev + c*fChannels + cd];
					}
					tmpW[k] = ptW[k][j];
				}
				delta[k] = tmpF - tmpM[k];
			}

			for ( k = 0; k < tmpK; k++ )
			{
				isCurrent[k] = false;
				if ( Mahalanobis(delta[k], tmpD[k]) < sqrt( cv::trace(tmpD[k]) ) )
				{
					count++;
					isCurrent[k] = true;
				}
				else
				{
					isCurrent[k] = false;
				}
			}

			if( count == 0 )
			{
				if ( tmpK < K )
				{
					tmpM[tmpK] = tmpF;
					tmpD[tmpK] = initDeviation*cv::Matx33d::eye();
					tmpW[tmpK] = alpha;
					tmpK = tmpK + 1;
				}
				else
				{
					tmpM[K-1] = tmpF;
					tmpD[K-1] = initDeviation*cv::Matx33d::eye();
					tmpW[K-1] = alpha;
				}
			}
			else
			{
				for ( k = 0; k < tmpK; k++ )
				{
					if ( isCurrent[k] )
					{
						w = (alpha / tmpW[k]);
						tmpM[k] += w * delta[k];
						tmpD[k] += std::min( 20*alpha, w ) * ( delta[k].t()*delta[k] );
					}
				}
			}
			w = 0;
			for ( k = 0; k < tmpK; k++ )
			{
				tmpW[k] = tmpW[k] * (1-alpha) + alpha*int(isCurrent[k]);
				w += tmpW[k];
			}

			for ( k = 0; k < tmpK; k++ )
			{
				tmpW[k] = tmpW[k] / w;
			}

			w = 0;
			for ( k = 0; k < tmpK; k++ )
			{
				w += tmpW[k];
			}

			bool noMov = false;
			while (!noMov)
			{
				noMov = true;
				for ( k = 0; k < tmpK-1; k++ )
				{
					if ( tmpW[k] < tmpW[k+1] )
					{
						w = tmpW[k];
						tmpW[k] = tmpW[k+1];
						tmpW[k+1] = w;
						delta[0] = tmpM[k];
						tmpM[k] = tmpM[k+1];
						tmpM[k+1] = delta[0];
						dhelp = tmpD[k];
						tmpD[k] = tmpD[k+1];
						tmpD[k+1] = dhelp;
						noMov = false;
					}
				}
			}

			count = -1;
			w = tmpW[0];

			for ( k = 1; k < tmpK; k++ )
			{
				if ( w >= (1-Cf) ) {
					count = k;
					break;
				}

				w += tmpW[k];
			}

			bool FG = sqrt(delta[0].dot(delta[0])) > T;

			if( FG )
			{
				ptMo[j] = 255;
			}

			ptK[j] = static_cast<uchar>(tmpK);
			for ( k = 0; k < tmpK; k++ )
			{
				for ( c = 0; c < fChannels; c++ )
				{
					ptM[k][iRGB + c] = tmpM[k](c);
					for ( cd = 0; cd < fChannels; cd++ )
					{
						ptD[k][iDev + c*fChannels + cd] = tmpD[k](c,cd);
					}
					ptW[k][j] = tmpW[k];
				}
			}
		}
	}
}

void GaussMixDetector::getMotionPicture( const cv::Mat& frame, cv::Mat& motion, bool cleanup )
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
	frame.convertTo( fClone, CV_MAKETYPE(CVType, fChannels) );

	// shouldn't algorithm be universal ??
	// i.e. you should be able to create the model for R and G channels only
	if (fChannels == 1)
	{
		getpwUpdateAndMotion(motion);
	}
	else if (fChannels == 3)
	{
		getpwUpdateAndMotionRGB(motion);
	}
	else
	{
		throw std::invalid_argument("Input image has non-standard number of channels.");
	}

	// optional erode + dilate processing of the motion image
	if (cleanup)
	{
		cv::erode(motion, motion, smCircle);
		cv::dilate(motion, motion, smCircle);
	}
}

#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "GaussMixDetector.h"


#ifndef LOG_TIME_ALL
	#define LOG_TIME_ALL FALSE
#endif


int main()
{
#if defined (LOG_TIME_ALL) && LOG_TIME_ALL
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTimeGauss = 0;
#endif

	bool camera = true;
	bool starter = true;
	bool perc = false;

	int history = 1000;
	double Cf = 0.1;
	GaussMixDetector bg(history, 20.0, 40, Cf);

	cv::Mat tmp, tmpx, tmpy, tmpLeft, tmpRight;
	cv::Mat frame;
	cv::Mat motion;

	std::string dataFolder("data/");
	std::string inputName(dataFolder + "input-test");
	std::string videoExt(".mp4");

	cv::VideoCapture cap;
	if (camera)
		cap.open(700);
	else
	{
		cap.open(inputName + videoExt);
		cap.set(cv::CAP_PROP_CONVERT_RGB, false);
		cap.set(cv::CAP_PROP_POS_FRAMES, 0);
	}
	if (!cap.isOpened())
	{
		std::cerr << "Failed to initiate video capture.";
		return 1;
	}

	cv::namedWindow("Frame");
	cv::namedWindow("Motion");

	cv::VideoWriter vidmotion;
	cv::VideoWriter vidsplit;

	unsigned int success = 0;
	unsigned int total = 0;

	for (; ; )
	{
		cap >> frame;							// read the source
		if (frame.empty()) break;				// check if reached the end

		// should test how it works with HLS and others
		//cv::cvtColor( frame, frame, CV_BGR2HLS );

#if defined (LOG_TIME_ALL) && LOG_TIME_ALL
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&t1);
#endif

		bg.getMotionPicture(frame, motion);		// get motion data from detector

#if defined (LOG_TIME_ALL) && LOG_TIME_ALL
		QueryPerformanceCounter(&t2);
		t1.QuadPart *= 1000;
		t2.QuadPart *= 1000;
		elapsedTimeGauss += (t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
#endif

		cv::imshow("Motion", motion);

		// Writing motion video
		if (cv::waitKey(15) == 3014656)   // delete key
		{
			starter = true;
		}
		if (starter)
		{
			if (!vidmotion.isOpened())
				vidmotion.open(inputName + "_motion" + videoExt, cv::VideoWriter::fourcc('P', 'I', 'M', '1'), 20, cv::Size(frame.cols, frame.rows), false);
			vidmotion << motion;
		}


		cv::imshow("Frame", frame);


		total++;

		if (cv::waitKey(15) == 32) break;  // space key to stop the loop
	}
#if defined(LOG_TIME_ALL) && LOG_TIME_ALL
	elapsedTimeGauss /= double(total);
#endif

	return 0;
}

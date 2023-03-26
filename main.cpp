#include "GaussMixDetector.h"

#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <tuple>

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>


cv::CommandLineParser
setup_cli_parser(int argc, char** argv)
{
	const cv::String keys =
		"{help h usage ? |      | print this message    }"
		"{@path          |      | input video file path }"
		"{history hist   |100   | history parameter     }"
		"{dev d          |20.0  | initial deviation     }"
		"{Cf cf          |0.2   | BF threshold          }"
		"{show preview   |      | enable preview        }";

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Gauss Mixture Motion Detector\n");

	return parser;
}

cv::String
trim_file_extension(const cv::String& filepath)
{
	return filepath.substr(0, filepath.find_last_of('.'));
}

using io_settings_t = std::tuple<bool, cv::String, cv::String>;

io_settings_t
get_io_settings(const cv::CommandLineParser& parser)
{
	const bool use_cam { !parser.has("@path") };

	const cv::String input_name {
		use_cam ? "cam" : parser.get<cv::String>("@path")
	};

	const cv::String video_ext { ".mp4" };
	const cv::String output_prefix { "_motion" };
	// TODO: add timestamp for cams
	const cv::String output_name { use_cam ? input_name + output_prefix + video_ext
		: trim_file_extension(
			cv::utils::fs::canonical(input_name)
			).append(output_prefix)
			.append(video_ext)
	};
	
	return { use_cam, input_name, output_name };
}

void try_open_camera(cv::VideoCapture& cap)
{
	// try a couple of camera indexes
	// (there's no solid method for finding cameras in OpenCV as of now)
	const std::array<int, 3> indices = { 0, 1, 700 };
	for (const auto& idx : indices)
	{
		if (cap.open(idx))
		{
			return;
		}
	}

	throw std::runtime_error("Failed to open cam capture");
}

void try_open_input_video_file(cv::VideoCapture& cap, const cv::String& inputName)
{
	cap.open(inputName);
	cap.set(cv::CAP_PROP_CONVERT_RGB, static_cast<double>(false));
	cap.set(cv::CAP_PROP_POS_FRAMES, 0);

	if (!cap.isOpened()) {
		throw std::runtime_error(
			std::string("Failed to open video capture from file ")
			.append(inputName)
		);
	}
}

void
try_open_output_video_file(
	cv::VideoWriter& vidmotion,
	const cv::String& outputName,
	double fps,
	const cv::Size& size)
{
	vidmotion.open(
		outputName,
		cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
		fps,
		size,
		false
	);

	if (!vidmotion.isOpened()) {
		throw std::runtime_error(
			std::string("Failed to open video output file ")
			.append(outputName)
		);
	}
}

void set_up_io(const io_settings_t& settings, cv::VideoCapture& input, cv::VideoWriter& output)
{
	const bool camera { std::get<0>(settings) };
	const cv::String& inputName { std::get<1>(settings) };

	if (camera)
	{
		try_open_camera(input);
	} else {
		try_open_input_video_file(input, inputName);
	}

	const cv::String& outputName { std::get<2>(settings) };
	try_open_output_video_file(
		output,
		outputName,
		camera ? 20.0 : input.get(cv::CAP_PROP_FPS),
		// FIXME: frame size from cam needs debugging!
		// MB pop a frame for size, cam-streamer has plenty
		cv::Size(
			static_cast<int>(input.get(cv::CAP_PROP_FRAME_WIDTH)),
			static_cast<int>(input.get(cv::CAP_PROP_FRAME_HEIGHT))
		)
	);
}

void main_loop(cv::VideoCapture& cap, cv::VideoWriter& vidmotion, GaussMixDetector& bg, bool showPreview)
{
#ifdef LOG_TIME_ALL
	unsigned int total = 0;
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTimeGauss = 0;
#endif

	// read - process - write cycle
	for (; ; )
	{
		cv::Mat frame {};
		cap >> frame;							// read the source

		if (frame.empty())
		{
			break;				// check if reached the end
		}

#ifdef LOG_TIME_ALL
		QueryPerformanceFrequency(&frequency);
		QueryPerformanceCounter(&t1);
#endif

		cv::Mat motion;
		bg.getMotionPicture(frame, motion);		// get motion data from detector

#ifdef LOG_TIME_ALL
		QueryPerformanceCounter(&t2);
		t1.QuadPart *= 1000;
		t2.QuadPart *= 1000;
		elapsedTimeGauss += (t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
#endif

		// Writing motion video
		vidmotion << motion;

		if (showPreview)
		{
			cv::imshow("Frame", frame);
			cv::imshow("Motion", motion);
		}

#ifdef LOG_TIME_ALL
		total++;
#endif

		if (cv::waitKey(15) == 32)
		{
			break;  // space key to stop the loop
		}
	}
#ifdef LOG_TIME_ALL
	elapsedTimeGauss /= double(total);
#endif
}

int main(int argc, char** argv)
{

	const cv::CommandLineParser parser { setup_cli_parser(argc, argv) };

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	const io_settings_t io_settings = get_io_settings(parser);
	cv::VideoCapture cap {};
	cv::VideoWriter vidmotion {};

	try {
		set_up_io(io_settings, cap, vidmotion);
	}
	catch (const std::runtime_error& e) {
		std::cerr << "Failed to initiate video input/output: " << e.what() << std::endl;
		return 1;
	}

	const bool showPreview { parser.has("show") };
	if (showPreview)
	{
		cv::namedWindow("Frame");
		cv::namedWindow("Motion");
	}

	const auto history { parser.get<int>("history") };
	const auto deviation { parser.get<double>("dev") };
	const auto threshold { parser.get<double>("Cf") };
	GaussMixDetector bg(history, threshold, deviation);

	main_loop(cap, vidmotion, bg, showPreview);

	return 0;
}

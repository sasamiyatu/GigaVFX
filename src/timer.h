#pragma once

#include <chrono>

struct Timer
{
	void tick();
	void tock();
	double get_elapsed_seconds();
	double get_elapsed_milliseconds();

	std::chrono::time_point<std::chrono::high_resolution_clock> begin;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;
};
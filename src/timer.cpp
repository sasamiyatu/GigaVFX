#include "timer.h"

void Timer::tick()
{
	begin = std::chrono::high_resolution_clock::now();
}

void Timer::tock()
{
	end = std::chrono::high_resolution_clock::now();
}

double Timer::get_elapsed_seconds()
{
	const std::chrono::duration<double> diff = end - begin;
	return diff.count();
}

double Timer::get_elapsed_milliseconds()
{
	std::chrono::duration<double, std::milli> ms_double = end - begin;
	return ms_double.count();
}

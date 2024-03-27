#include "Timer.h"

Timer::Timer()
{
	Start();
	End();
	m_measure_error = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start);
}

void Timer::Start()
{
	m_start = std::chrono::high_resolution_clock::now();
}

void Timer::End()
{
	m_end = std::chrono::high_resolution_clock::now();
}

std::chrono::microseconds Timer::GetTime()
{
	std::chrono::microseconds time = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start) - m_measure_error;
	return time;
}


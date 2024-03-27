#pragma once
#include <chrono>
#include <iostream>
class Timer
{
public:
	Timer();
	void Start();
	void End();
	std::chrono::microseconds GetTime();
private:
	std::chrono::steady_clock::time_point m_start;
	std::chrono::steady_clock::time_point m_end;
	std::chrono::microseconds m_measure_error;
};

inline std::ostream& operator<<(std::ostream& os, Timer& obj)
{
	return os << "Time is " << obj.GetTime() << std::endl;
}
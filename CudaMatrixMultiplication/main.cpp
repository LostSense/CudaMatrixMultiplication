#include <iostream>
#include "CUAdditor.h"

int main(int argc, char** argv)
{
	std::cout << "Hello this cuda world" << std::endl;
	CUAdditor obj;
	obj.Run();
	obj.Run2();

	return 0;
}

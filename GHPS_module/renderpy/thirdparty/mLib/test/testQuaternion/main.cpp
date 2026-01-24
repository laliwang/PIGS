#include <iostream>

#include "mLibCore.h"
#include "mLibCore.cpp"
using namespace ml;

int main() {
    quatf q(0.5, 0.5, 0, 0);
    std::cout << "Original quaternion: " << q << std::endl;

    // Save to file
    std::ofstream ofs("saved_quat.txt", std::ios::out);
    if (!ofs) {
        return false;
    }
    ofs << q;

    // Read from file
    std::ifstream ifs("saved_quat.txt", std::ios::in);
    if (!ifs) {
        return false;
    }
    ifs >> q;
    std::cout << "Read quaternion: " << q << std::endl;
    return 0;
}

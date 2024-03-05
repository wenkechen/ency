#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    std::ifstream input_file("input.txt");
    std::ofstream output_file("output_file.txt");

    if (!input_file.is_open() || !output_file.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(input_file, line)) {
        output_file << "cwk -> " <<  line << std::endl;
    }
    input_file.close();
    output_file.close();

    std::cout << "File read and write operation completed." << std::endl;
    return 0;
}

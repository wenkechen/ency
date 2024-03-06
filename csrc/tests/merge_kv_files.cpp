/**
 *  Given N large files, each of which is X, and stores a certain amount of data in the form of (k,v), and there are also M small files, each of 
 *  which is Y, and also stores data in the form of (k, v'). Now please de-duplicate and merge these two kinds of files. That is, for the same key,
 *  duplicate values are filtered and different values are retained.
 *	
 *	For example, N = 1 ; X = 1TB; M = 1,000,000,000; Y = 1KB
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

void show_pv(const std::vector<std::pair<std::string, std::string>>& pv) {
    for (const auto& kv : pv) {
        std::cout << std::get<0>(kv) << " " << std::get<1>(kv) << std::endl;
    }
}

std::string assembly_file_name(const std::string& file_name, const std::string& suffix, int64_t idx, char delimiter) {
    std::stringstream ss(file_name);
    std::stringstream ssa;
    std::string token;
    std::getline(ss, token, delimiter);
    ssa << token << suffix << idx << delimiter;
    std::getline(ss, token, '.');
    ssa << token;
    return ssa.str();
}

std::string assembly_file_name(const std::string& prefix, int64_t idx, const std::string& extension) {
    std::stringstream ssa;
    ssa << prefix << idx << '.' << extension;
    return ssa.str();
}

auto comp = [](const std::pair<std::string, std::string>& lhs, const std::pair<std::string, std::string>& rhs) {
    if (lhs.first < rhs.first) {
        return true;
    } else if (lhs.first > rhs.first) {
        return false;
    } else {
        return lhs.second < rhs.second;
    }
};

int partition(std::vector<std::pair<std::string, std::string>>& arr, int low, int high) {
    std::pair<std::string, std::string> pivot = arr[low];
    int i = low + 1, j = high;
    while (i < j) {
        while (i <= high && comp(arr[i], pivot)) ++i;
        while (j > low && !comp(arr[j], pivot)) --j;
        if (i < j) std::swap(arr[i], arr[j]);
    }
    if (!comp(arr[low], arr[j])) std::swap(arr[low], arr[j]);
    return j;
}

void doQuickSort(std::vector<std::pair<std::string, std::string>>& arr, int low, int high) {
    if (low < high) {
        int mid = partition(arr, low, high);
        doQuickSort(arr, low, mid - 1);
        doQuickSort(arr, mid + 1, high);
    }
}

void quick_sort(std::vector<std::pair<std::string, std::string>>& pv) {
    // std::sort(pv.begin(), pv.end(), [](const std::pair<std::string, std::string>& lhs, const std::pair<std::string, std::string>& rhs) {
    //     if (lhs.first < rhs.first) {
    //         return true;
    //     } else if (lhs.first > rhs.first) {
    //         return false;
    //     } else {
    //         return lhs.second < rhs.second;
    //     }
    // });

    // std::sort(pv.begin(), pv.end(), comp);

    doQuickSort(pv, 0, pv.size() - 1);
}

int64_t parse_buffer(char* buffer, int buf_sz, std::vector<std::pair<std::string, std::string>>& pv) {
    assert(buf_sz >= 2 && buffer[0] == '(');
    int64_t slow_idx = 0, fast_idx = 0;
    int64_t state = 0, key_idx = -1, value_idx = -1;
    std::string key, value;
    while (fast_idx < buf_sz) {
        char ch = buffer[fast_idx];
        switch (ch) {
        case '(':
            state = 1;  // enter k state
            key_idx = -1;
            break;
        case ',':
            state = 2;  // enter v state
            key = std::string(buffer + key_idx, fast_idx - key_idx);
            value_idx = -1;
            break;
        case ')':
            state = 3;  // enter e state
            value = std::string(buffer + value_idx, fast_idx - value_idx);
            pv.emplace_back(key, value);
            slow_idx = fast_idx;
            break;
        default:
            if (state == 1) {
                if (key_idx == -1) {
                    key_idx = fast_idx;
                }
            } else if (state == 2) {
                if (value_idx == -1) {
                    value_idx = fast_idx;
                }
            } else if (state == 3) {
                state = 0;  // enter s state again
            }
            break;
        }
        ++fast_idx;
    }

    return slow_idx + 1;
}

void dump_sorted_kvs_into_file(std::vector<std::pair<std::string, std::string>>& pv, const std::string& fname) {
    std::ofstream output_file(fname);

    if (!output_file.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        return;
    }

    for (const auto& kv : pv) {
        output_file << '(' << kv.first << ',' << kv.second << ')';
    }

    std::cout << "dumped sorted kvs into " << fname << std::endl;
}

std::vector<std::string> split_large_file(std::string file_name, int64_t chunk_size, int64_t start_pos) {
    assert(chunk_size == 150 && start_pos == 0);
    std::vector<std::string> splited_large_files;
    int64_t pos = start_pos;
    int64_t split_idx = 0;  // the idx of current splited large file

    // open large file
    std::ifstream large_file(file_name);
    if (!large_file.is_open()) {
        std::cerr << "Error opening " << file_name << std::endl;
        return splited_large_files;
    }

    // grab the length of large file
    large_file.seekg(0, large_file.end);
    int64_t length = large_file.tellg();

    // reset the position of the next char
    large_file.seekg(0);
    while (pos < length - 2) {
        int64_t curr_chunk_sz = std::min(length - pos, chunk_size);
        char* buffer = new char[curr_chunk_sz + 1];
        large_file.read(buffer, curr_chunk_sz);
        buffer[curr_chunk_sz] = 0;
        std::vector<std::pair<std::string, std::string>> pv;
        pos += parse_buffer(buffer, curr_chunk_sz, pv);
        delete[] buffer;
        quick_sort(pv);
        dump_sorted_kvs_into_file(pv, assembly_file_name(file_name, "_split_", split_idx, '.'));
        large_file.seekg(pos);
        ++split_idx;
    }

    large_file.close();

    return splited_large_files;
}


std::vector<std::string> merge_small_files(std::vector<std::string> file_names, int64_t chunk_size, const std::string& prefix) {
	std::vector<std::string> merged_files;
    std::vector<std::pair<std::string, std::string>> pv;
	int64_t merged_idx = 0; // the idx of current merged file
	std::string merged_file_name = assembly_file_name(prefix, merged_idx, "txt");
	
    // open merged file
    std::ifstream merged_file(merged_file_name);
    if (!merged_file.is_open()) {
        std::cerr << "Error opening " << merged_file_name << std::endl;
        return merged_files;
    }



	
	
	int64_t pos = start_pos;
    int64_t split_idx = 0;  // the idx of current splited large file

    // open large file
    std::ifstream large_file(file_name);
    if (!large_file.is_open()) {
        std::cerr << "Error opening " << file_name << std::endl;
        return splited_large_files;
    }

    // grab the length of large file
    large_file.seekg(0, large_file.end);
    int64_t length = large_file.tellg();

    // reset the position of the next char
    large_file.seekg(0);
    while (pos < length - 2) {
        int64_t curr_chunk_sz = std::min(length - pos, chunk_size);
        char* buffer = new char[curr_chunk_sz + 1];
        large_file.read(buffer, curr_chunk_sz);
        buffer[curr_chunk_sz] = 0;
        std::vector<std::pair<std::string, std::string>> pv;
        pos += parse_buffer(buffer, curr_chunk_sz, pv);
        delete[] buffer;
        quick_sort(pv);
        dump_sorted_kvs_into_file(pv, assembly_file_name(file_name, "_split_", split_idx, '.'));
        large_file.seekg(pos);
        ++split_idx;
    }

    large_file.close();

    return splited_large_files;
}

int main() {
    const std::string large_file_name = "/root/data/original_large_file.txt";
	std::vector<std::string> to_be_merged_files;
    auto splited_large_files = split_large_file(large_file_name, 150, 0);
	to_be_merged_files.insert(to_be_merged_files.end(), splited_large_files.begin(),splited_large_files.end());
    return 0;
}

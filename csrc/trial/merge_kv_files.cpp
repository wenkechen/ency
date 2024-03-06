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
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "thread_pool.h"

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

auto comparePair = [](const std::pair<std::string, std::string>& lhs, const std::pair<std::string, std::string>& rhs) {
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
        while (i <= high && comparePair(arr[i], pivot)) ++i;
        while (j > low && !comparePair(arr[j], pivot)) --j;
        if (i < j) std::swap(arr[i], arr[j]);
    }
    if (!comparePair(arr[low], arr[j])) std::swap(arr[low], arr[j]);
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

void write_into_intermediate_file(std::vector<std::pair<std::string, std::string>>& pv, const std::string& fname) {
    std::ofstream intermediate_file(fname);

    if (!intermediate_file.is_open()) {
        std::cerr << "Error opening " << fname << std::endl;
        return;
    }

    for (const auto& kv : pv) {
        intermediate_file << '(' << kv.first << ',' << kv.second << ')';
    }

    intermediate_file.close();

    std::cout << "write into " << fname << std::endl;
}

std::vector<std::string> split_large_file(std::string file_name, int64_t chunk_size, int64_t start_pos) {
    assert(chunk_size == 150 && start_pos == 0);
    std::vector<std::string> splited_files;
    int64_t pos = start_pos;
    int64_t splited_idx = 0;  // the idx of current splited large file

    // open large file
    std::ifstream large_file(file_name);
    if (!large_file.is_open()) {
        std::cerr << "Error opening " << file_name << std::endl;
        return splited_files;
    }

    // grab the length of large file
    large_file.seekg(0, large_file.end);
    int64_t length = large_file.tellg();

    // reset the position of the next char
    large_file.seekg(0);
    while (pos < length - 2) {
        int64_t cur_chunk_sz = std::min(length - pos, chunk_size);
        char* buffer = new char[cur_chunk_sz + 1];
        large_file.read(buffer, cur_chunk_sz);
        buffer[cur_chunk_sz] = 0;
        std::vector<std::pair<std::string, std::string>> pv;
        pos += parse_buffer(buffer, cur_chunk_sz, pv);
        delete[] buffer;
        quick_sort(pv);
        std::string splited_file_name = assembly_file_name(file_name, "_split_", splited_idx, '.');
        write_into_intermediate_file(pv, splited_file_name);
        splited_files.emplace_back(splited_file_name);
        large_file.seekg(pos);
        ++splited_idx;
    }

    large_file.close();

    return splited_files;
}

std::vector<std::string> merge_small_files(std::vector<std::string> file_names, int64_t chunk_size, const std::string& prefix) {
    std::vector<std::string> merged_files;
    std::vector<std::pair<std::string, std::string>> pv;
    int64_t quota = chunk_size;
    int64_t merged_idx = 0;  // the idx of current merged file
    std::string merged_file_name = assembly_file_name(prefix, merged_idx, "txt");

    for (const auto& file_name : file_names) {
        std::ifstream small_file(file_name);
        // grab the length of small file
        small_file.seekg(0, small_file.end);
        int64_t length = small_file.tellg();
        small_file.seekg(0);
        char* buffer = new char[length + 1];
        small_file.read(buffer, length);
        buffer[length] = 0;
        int64_t cur_chunk_sz = std::min(length, quota);
        int64_t consumed_quota = parse_buffer(buffer, cur_chunk_sz, pv);
        quota -= consumed_quota;

        if (cur_chunk_sz < length) {
            quick_sort(pv);
            write_into_intermediate_file(pv, merged_file_name);
            merged_files.emplace_back(merged_file_name);
            pv.clear();

            // setup next merged file
            quota = chunk_size;
            ++merged_idx;
            merged_file_name = assembly_file_name(prefix, merged_idx, "txt");
            quota -= parse_buffer(buffer + consumed_quota, cur_chunk_sz - consumed_quota, pv);
        }

        delete[] buffer;
        small_file.close();
    }

    if (!pv.empty()) {
        quick_sort(pv);
        write_into_intermediate_file(pv, merged_file_name);
        merged_files.emplace_back(merged_file_name);
        pv.clear();
    } else {
        std::remove(merged_file_name.c_str());
    }

    return merged_files;
}

struct TupleCompare {
    bool operator()(const std::tuple<std::string, std::string, int64_t>& lhs, const std::tuple<std::string, std::string, int64_t>& rhs) const {
        // Compare the first element of the tuples
        if (std::get<0>(lhs) != std::get<0>(rhs)) {
            return std::get<0>(lhs) > std::get<0>(rhs);  // Compare integers in descending order
        }

        // Compare the second element of the tuples
        if (std::get<1>(lhs) != std::get<1>(rhs)) {
            return std::get<1>(lhs) > std::get<1>(rhs);  // Compare characters in descending order
        }

        // Compare the third element of the tuples
        return std::get<2>(lhs) > std::get<2>(rhs);  // Compare doubles in descending order
    }
};

auto equalTuple = [](const std::tuple<std::string, std::string, int64_t>& lhs, const std::tuple<std::string, std::string, int64_t>& rhs) {
    return (std::get<0>(lhs) == std::get<0>(rhs)) && (std::get<1>(lhs) == std::get<1>(rhs));
};

void readOneKV(
    std::priority_queue<std::tuple<std::string, std::string, int64_t>, std::vector<std::tuple<std::string, std::string, int64_t>>, TupleCompare>& pq,
    std::vector<std::ifstream>& fv,
    int64_t fidx) {
    bool done = false;
    while (!done) {
        int64_t state = 0;
        std::stringstream keySS, valueSS;
        std::string key, value;
        char ch;
        while (state != 3 && fv[fidx].get(ch)) {
            switch (ch) {
            case '(':
                state = 1;  // enter k state
                break;
            case ',':
                key = keySS.str();
                state = 2;  // enter v state
                break;
            case ')':
                value = valueSS.str();
                state = 3;  // enter e state
                break;
            default:
                if (state == 1) {
                    keySS << ch;
                } else if (state == 2) {
                    valueSS << ch;
                } else if (state == 3) {
                    state = 0;  // enter s state again
                }
                break;
            }
        }

        if (state == 3) {
            if (pq.empty()) {
                pq.emplace(key, value, fidx);
                done = true;
            } else {
                if (equalTuple(pq.top(), {key, value, fidx})) {
                    // drop it.
                } else {
                    pq.emplace(key, value, fidx);
                    done = true;
                }
            }
        }

        if (fv[fidx].eof()) {
            fv[fidx].close();
            done = true;
        }
    }
}

void writeOneKV(
    std::ofstream& ofs,
    std::priority_queue<std::tuple<std::string, std::string, int64_t>, std::vector<std::tuple<std::string, std::string, int64_t>>, TupleCompare>& pq,
    std::vector<std::ifstream>& fv,
    int64_t buffer_size,
    int64_t& quota) {
    auto smallest_elem = pq.top();
    pq.pop();
    const std::string key = std::get<0>(smallest_elem);
    const std::string value = std::get<1>(smallest_elem);
    const int64_t fidx = std::get<2>(smallest_elem);
    int64_t consumed_quota = key.size() + value.size() + 3;  // '(' key ',' value' )'
    if (quota < consumed_quota) {
        ofs.flush();
        quota = buffer_size;
    }
    ofs << '(' << key << '.' << value << ')';
    quota -= consumed_quota;
    readOneKV(pq, fv, fidx);
}

void merge_intermediate_files(const std::vector<std::string>& intermediate_files, const std::string& output_file_name, int64_t buffer_size) {
    std::ofstream output_file(output_file_name);

    if (!output_file.is_open()) {
        std::cerr << "Error opening " << output_file_name << std::endl;
        return;
    }

    std::vector<std::ifstream> fv;
    for (const auto& file_name : intermediate_files) {
        fv.emplace_back(file_name);
    }

    std::priority_queue<std::tuple<std::string, std::string, int64_t>, std::vector<std::tuple<std::string, std::string, int64_t>>, TupleCompare> pq;

    for (int i = 0; i < fv.size(); ++i) {
        if (fv[i].is_open()) {
            readOneKV(pq, fv, i);
        }
    }

    int64_t quota = buffer_size;
    while (!pq.empty()) {
        writeOneKV(output_file, pq, fv, buffer_size, quota);
    }

    output_file.close();
}

int main() {
    const int64_t chunk_size = 150;
    const std::string large_file_name = "/root/data/original_large_file.txt";
    std::vector<std::string> intermediate_files;

    std::vector<std::string> small_file_names = {
        "/root/data/original_small_file_0.txt",
        "/root/data/original_small_file_1.txt",
        "/root/data/original_small_file_2.txt",
        "/root/data/original_small_file_3.txt",
    };
    const std::string prefix = "/root/data/original_small_file_merged_";

    auto splited_files = split_large_file(large_file_name, chunk_size, 0);
    auto merged_files = merge_small_files(small_file_names, chunk_size, prefix);

    // trial::ThreadPool pool;
    // std::future<std::vector<std::string> > split_large_file_fut = pool.execute(split_large_file, large_file_name, chunk_size, 0);
    // std::future<std::vector<std::string> > merge_small_files_fut = pool.execute(merge_small_files, small_file_names, chunk_size, prefix);
    // auto splited_files = split_large_file_fut.get();
    // auto merged_files = merge_small_files_fut.get();

    intermediate_files.insert(intermediate_files.end(), splited_files.begin(), splited_files.end());
    intermediate_files.insert(intermediate_files.end(), merged_files.begin(), merged_files.end());

    // TODO: merge intermediate files into larger files

    const std::string output_file_name = "/root/data/output_file.txt";
    const int64_t buffer_size = 256;
    merge_intermediate_files(intermediate_files, output_file_name, buffer_size);

    return 0;
}

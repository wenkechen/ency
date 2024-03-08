#include "thread_pool.h"

namespace trial {
ThreadPool::ThreadPool(size_t nr_threads) {
    for (size_t i = 0; i < nr_threads; ++i) {
        // start waiting threads. Workers listen for changes through
        // the ThreadPool member condition_variable
        _threads.emplace_back(std::thread([&]() {
            std::unique_lock<std::mutex> queue_lock(_task_mutex, std::defer_lock);
            while (true) {
                queue_lock.lock();
                _task_cv.wait(queue_lock, [&]() -> bool {
                    return !_tasks.empty() || _stop_threads;
                });

                // used by dtor to stop all threads without having to unceremoniously stop tasks.
                // The tasks must all be finished, lest we break a promise and risk a `future`
                // object throwing an exception.
                if (_stop_threads && _tasks.empty()) {
                    return;
                }

                // to initialize tmp_task, we must move the unique_ptr from the queue to the local stack.
                // Since a unique_ptr cannot be copied (obviously), it must be explicitly moved. This
                // transfers ownership of the pointed-to object to *this.
                auto tmp_task = std::move(_tasks.front());
                _tasks.pop();
                queue_lock.unlock();
                (*tmp_task)();
            }
        }));
    }
}

ThreadPool::~ThreadPool() {
    _stop_threads = true;
    _task_cv.notify_all();
    for (std::thread& thread : _threads) {
        thread.join();
    }
}
}  // namespace trial

// #include <iostream>
// int multiply(int x, int y) {
//     return x * y;
// }

// int main() {
//     trial::ThreadPool pool(4);
//     std::vector<std::future<int>> futs;
//     for (const int& x : {2, 4, 7, 13}) {
//         futs.push_back(pool.execute(multiply, x, 2));
//     }
//     for (auto& fut : futs) {
//         std::cout << fut.get() << std::endl;
//     }
//     return 0;
// }
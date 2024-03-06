#pragma once
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace trial {
class ThreadPool {
public:
    ThreadPool(size_t nr_threads = std::thread::hardware_concurrency());
    // since std::thread objects are not copiable, it doesn't make sense for a ThreadPool to be copiable.
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool();

    template <typename F, typename... Args, std::enable_if_t<std::is_invocable_v<F&&, Args&&...>, int> = 0>
    auto execute(F&&, Args&&...);

private:
    // _TaskContainerBase exists only to serve as an abstract base for __TaskContainer.
    class _TaskContainerBase {
    public:
        virtual ~_TaskContainerBase(){};
        virtual void operator()() = 0;
    };

    // _TaskContainer takes a typename F, which must be Callable and MoveConstructible.
    // Furthermore, F must be callable with no argument. It can, for example, be a bind object with
    // no placeholders. F may or may not be CopyConstructible.
    template <typename F, std::enable_if_t<std::is_invocable_v<F&&>, int> = 0>
    class _TaskContainer : public _TaskContainerBase {
    public:
        // here, std::forward is needed because we need the construction of _f not
        // to bind an lvalue reference - it is not a guarantee that an object of
        // type F is CopyConstructible, only that it is MoveConstructible.
        _TaskContainer(F&& func) : _f(std::forward<F>(func)) {}
        void operator()() override { _f(); };

    private:
        F _f;
    };

    template <typename F>
    _TaskContainer(F) -> _TaskContainer<std::decay<F>>;

    std::vector<std::thread>
        _threads;
    std::mutex _task_mutex;
    std::condition_variable _task_cv;
    std::queue<std::unique_ptr<_TaskContainerBase>> _tasks;
    bool _stop_threads = false;
};

template <typename F, typename... Args, std::enable_if_t<std::is_invocable_v<F&&, Args&&...>, int>>
auto ThreadPool::execute(F&& function, Args&&... args) {
    std::unique_lock<std::mutex> queue_lock(_task_mutex, std::defer_lock);
    std::packaged_task<std::invoke_result_t<F, Args...>()> pkg_task(
        [_f = std::move(function), _fargs = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            return std::apply(std::move(_f), std::move(_fargs));
        });
    std::future<std::invoke_result_t<F, Args...>> fut = pkg_task.get_future();
    queue_lock.lock();
    // this lambda move-captures the packaged_task declared above. Since the packaged_task type is not CopyConstructible,
    // the function is not CopyConstructible either - hence the need for a _task_container to wrap around it.
    _tasks.emplace(std::unique_ptr<_TaskContainerBase>(new _TaskContainer([task(std::move(pkg_task))]() mutable { task(); })));
    queue_lock.unlock();
    _task_cv.notify_one();
    return std::move(fut);
}

}  // namespace trial

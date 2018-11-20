#include "parallel.h"
#include "stats.h"
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <cstdio>
#include <condition_variable>
#include <functional>
#include <cmath>
#include <cstdio>
#include <algorithm>

//logger https://github.com/gabime/spdlog
//format syntax  http://fmtlib.net/latest/syntax.html
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

class ParallelForLoop; // forward declare.

static bool shutdowThread = false;

static std::vector<std::thread> threads;

static std::mutex workListMutex;
static std::condition_variable workListCondition;

static ParallelForLoop *workList = nullptr;

thread_local int threadIndex;

// Bookkeeping variables to help with the implementation of
// MergeWorkerThreadStats().
static std::atomic<bool> reportWorkerStats{ false };

// Number of workers that still need to report their stats.
static std::atomic<int> reporterCount;

// After kicking the workers to report their stats, the main thread waits
// on this condition variable until they've all done so.
static std::condition_variable reportDoneCondition;
static std::mutex reportDoneMutex;

// Used for log
static std::shared_ptr<spdlog::logger> mylogger = nullptr;

static int numSystemCores(void)
{
    return std::max(1u, std::thread::hardware_concurrency());
}

static int maxThreadIndex(void)
{
    return numSystemCores();
}

// Abstract the job partition
class ParallelForLoop
{
public:
    ParallelForLoop(std::function<void(int64_t)> func, int64_t maxIndex, int chunkSize)
        :mFunc(func)
        ,mMaxIndex(maxIndex)
        ,mChunkSize(chunkSize)
	{
        mylogger->info("ParallelForLoop, maxindex={} mChunkSize={} Constructed.", mMaxIndex, mChunkSize);
	}

public:

    const int64_t mMaxIndex;
    const int mChunkSize;

    int64_t mNextIndex = 0;
    int mActiveWorkers = 0;
    std::function<void(int64_t)> mFunc;
    ParallelForLoop *next = nullptr;

public:
    bool Finished(void) const
    {
        return mNextIndex >= mMaxIndex && mActiveWorkers == 0;
    }

};

// worker thread
void workerThread(int tIndex)
{
    mylogger->info("Thread {} starting.", tIndex);
    threadIndex = tIndex; // thread local
    
    std::unique_lock<std::mutex> lock(workListMutex);
    while (!shutdowThread)
    {
		if (reportWorkerStats) //atomic
		{
			ReportThreadStats(); 
			if (--reporterCount == 0)
				// Once all worker threads have merged their stats, wake up main thread.
				reportDoneCondition.notify_one();
			// Now sleep again.
			workListCondition.wait(lock);
		}
       else if (!workList) 
       {
           workListCondition.wait(lock);
       }
       else
       {
            // Thread wakes and does job here.
            ParallelForLoop &loop = *workList;
            int64_t indexStart = loop.mNextIndex;
            int64_t indexEnd   = std::min(indexStart + loop.mChunkSize, loop.mMaxIndex);
            loop.mNextIndex = indexEnd;
            if (loop.mNextIndex == loop.mMaxIndex) workList = loop.next;
            loop.mActiveWorkers++;

            lock.unlock();

            mylogger->info("Thread {} starts work on [{}, {}].", threadIndex, indexStart, indexEnd);
            for (uint64_t i = indexStart; i < indexEnd; ++i)
            {
                if (loop.mFunc) loop.mFunc(i);
            }
            lock.lock();
            loop.mActiveWorkers--;
            if (loop.Finished()) workListCondition.notify_all();
       }
    }
    mylogger->info("Thread {} exiting.", tIndex);
}

// PUBLIC interface
// Called by Main Thread
// count: total work in a ParallelForLoop
// chunkSize:  work for a single worker in a ParallelLoop
void ParallelFor(std::function<void(int64_t)> func, int64_t count, int chunkSize)
{
    // Parallel is not available or no need.
   if (threads.size() == 0 || count < chunkSize) 
   {
       for (int64_t i = 0; i < count; ++i) func(i);
       return;
   }

   // Create ParallelFor object, Add it into work list.
   ParallelForLoop loop(std::move(func), count, chunkSize);
   workListMutex.lock();
   loop.next = workList;
   workList = &loop;
   workListMutex.unlock();

   std::unique_lock<std::mutex> lock(workListMutex);
   workListCondition.notify_all();

   while (!loop.Finished())
   {
        int64_t indexStart = loop.mNextIndex;
        int64_t indexEnd   = std::min(indexStart + loop.mChunkSize, loop.mMaxIndex);
        loop.mNextIndex = indexEnd;
        if (loop.mNextIndex == loop.mMaxIndex) workList = loop.next;
        loop.mActiveWorkers++;

        lock.unlock(); // release lock after update loop parameter.
        
        mylogger->info("Main Thread starts work on [{}, {}].", indexStart, indexEnd);
        for (uint64_t i = indexStart; i < indexEnd; ++i)
        {
            if (loop.mFunc) loop.mFunc(i);
        }
        lock.lock();
        loop.mActiveWorkers--;
   }
}



//#define SPDLOG_DEBUG_TEST

// Called in Main Thread
void parallelInit(void)
{
#ifdef SPDLOG_DEBUG_TEST
    spdlog::info("Welcome to spdlog version {}.{}.{} !", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
    spdlog::warn("Easy padding in numbers like {:08d}", 12);
    spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    spdlog::info("Support for floats {:03.2f}", 1.23456);
    spdlog::info("Positional args are {1} {0}..", "too", "supported");
    spdlog::info("{:>8} aligned, {:>8} aligned", "right", "left");
#endif
    try
    {
        if (!mylogger) mylogger = spdlog::basic_logger_mt("mylogger", "basic-log.txt");
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
        spdlog::critical("Log init failed: {0:s};", ex.what());
        system("pause");
        exit(1);
    }

    if (threads.size() > 0) return;
    int nThreads = maxThreadIndex();
    threadIndex = 0; // thread local variable.
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(workerThread, i+1));
    mylogger->info("Spaw {} worker threads.", nThreads - 1);
    mylogger->info("Parallel module init Ok.");
}

// Called in Main Thread
// Main Thread would wait for all worker to exit.
void parallelCleanup(void)
{
    if (threads.size() == 0) return;

    {
        std::lock_guard<std::mutex> lock(workListMutex);
        shutdowThread = true;
        workListCondition.notify_all();
    }

    for (std::thread &thread : threads) thread.join();
    threads.erase(threads.begin(), threads.end());
    shutdowThread = false;
    mylogger->info("Parallel module cleanup Ok.");
}

// Called in Main Thread
void MergeWorkerThreadStats()
{
	std::unique_lock<std::mutex> lock(workListMutex);
	std::unique_lock<std::mutex> doneLock(reportDoneMutex);
	// Set up state so that the worker threads will know that we would like
	// them to report their thread-specific stats when they wake up.
	reportWorkerStats = true;
	reporterCount = threads.size();

	// Wake up the worker threads.
	workListCondition.notify_all();

	// Wait for all of them to merge their stats.
	reportDoneCondition.wait(lock, []() { return reporterCount == 0; });

	reportWorkerStats = false;
}



//#define LOCAL_TEST
#ifdef LOCAL_TEST
#include <assert.h>
#include "progressreporter.h"
int main()
{
#define N (1024 * 1024 * 1)
    static int64_t a[N] = {0};
    parallelInit();

    // Main Thread also does jobs
    {
        ProgressReporter reporter(N, "Progressing");
        ParallelFor([&](int64_t i) {
             a[i] = i;
             reporter.update();
        }, N, 4096);
        reporter.done();
    }

    MergeWorkerThreadStats();
    ReportThreadStats();
    PrintStats(stdout);
    ClearStats();
    parallelCleanup();

    for (int64_t i = 0; i < N; ++i)
        assert(a[i] == i);

    system("pause");
    return 0;
}

#endif
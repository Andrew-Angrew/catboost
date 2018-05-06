#include <library/threading/future/future.h>
#include <library/threading/local_executor/local_executor.h>
#include <library/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/system/mutex.h>
#include <util/system/rwlock.h>
#include <util/system/thread.h>
#include <util/thread/pool.h>

using namespace NPar;

class TTestException: public yexception {
};

static const int DefaultThreadsCount = 41;
static const int DefaultRangeSize = 999;

Y_UNIT_TEST_SUITE(ExecRangeWithFutures){
    bool AllOf(const TVector<int>& vec, int value){
        return AllOf(vec, [value](int element) { return value == element; });
}

void AsyncRunAndWaitFuturesReady(int rangeSize, int threads) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threads);
    TAtomic signal = 0;
    TVector<int> data(rangeSize, 0);
    TVector<NThreading::TFuture<void>> futures = localExecutor.ExecRangeWithFutures([&signal, &data](int i) {
        UNIT_ASSERT(data[i] == 0);
        while (AtomicGet(signal) == 0)
            ;
        data[i] += 1;
    },
                                                                                    0, rangeSize, TLocalExecutor::HIGH_PRIORITY);
    UNIT_ASSERT(AllOf(data, 0));
    for (auto& future : futures)
        UNIT_ASSERT(!future.HasValue());
    AtomicSet(signal, 1);
    for (auto& future : futures) {
        future.GetValueSync();
    }
    UNIT_ASSERT(AllOf(data, 1));
}

Y_UNIT_TEST(AsyncRunRangeAndWaitFuturesReady) {
    AsyncRunAndWaitFuturesReady(DefaultRangeSize, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunOneTaskAndWaitFuturesReady) {
    AsyncRunAndWaitFuturesReady(1, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunRangeAndWaitFuturesReadyOneExtraThread) {
    AsyncRunAndWaitFuturesReady(DefaultRangeSize, 1);
}

Y_UNIT_TEST(AsyncRunOneThreadAndWaitFuturesReadyOneExtraThread) {
    AsyncRunAndWaitFuturesReady(1, 1);
}

Y_UNIT_TEST(AsyncRunTwoRangesAndWaitFuturesReady) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(DefaultThreadsCount);
    TAtomic signal = 0;
    TVector<int> data1(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures1 = localExecutor.ExecRangeWithFutures([&signal, &data1](int i) {
        UNIT_ASSERT(data1[i] == 0);
        while (AtomicGet(signal) == 0)
            ;
        data1[i] += 1;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::HIGH_PRIORITY);
    TVector<int> data2(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures2 = localExecutor.ExecRangeWithFutures([&signal, &data2](int i) {
        UNIT_ASSERT(data2[i] == 0);
        while (AtomicGet(signal) == 0)
            ;
        data2[i] += 2;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::HIGH_PRIORITY);
    UNIT_ASSERT(AllOf(data1, 0));
    UNIT_ASSERT(AllOf(data2, 0));
    AtomicSet(signal, 1);
    for (int i = 0; i < DefaultRangeSize; ++i) {
        futures1[i].GetValueSync();
        futures2[i].GetValueSync();
    }
    UNIT_ASSERT(AllOf(data1, 1));
    UNIT_ASSERT(AllOf(data2, 2));
}

void AsyncRunRangeAndWaitExceptions(int rangeSize, int threadsCount) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadsCount);
    TAtomic signal = 0;
    TVector<int> data(rangeSize, 0);
    TVector<NThreading::TFuture<void>> futures = localExecutor.ExecRangeWithFutures([&signal, &data](int i) {
        UNIT_ASSERT(data[i] == 0);
        while (AtomicGet(signal) == 0)
            ;
        data[i] += 1;
        throw 10000 + i;
    },
                                                                                    0, rangeSize, TLocalExecutor::HIGH_PRIORITY);
    UNIT_ASSERT(AllOf(data, 0));
    UNIT_ASSERT(futures.ysize() == rangeSize);
    AtomicSet(signal, 1);
    int exceptionsCaught = 0;
    for (int i = 0; i < rangeSize; ++i) {
        try {
            futures[i].GetValueSync();
        } catch (int& e) {
            if (e == 10000 + i) {
                ++exceptionsCaught;
            }
        }
    }
    UNIT_ASSERT(exceptionsCaught == rangeSize);
    UNIT_ASSERT(AllOf(data, 1));
}

Y_UNIT_TEST(AsyncRunRangeAndWaitExceptions) {
    AsyncRunRangeAndWaitExceptions(DefaultRangeSize, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunOneTaskAndWaitExceptions) {
    AsyncRunRangeAndWaitExceptions(1, DefaultThreadsCount);
}

Y_UNIT_TEST(AsyncRunRangeAndWaitExceptionsOneExtraThread) {
    AsyncRunRangeAndWaitExceptions(DefaultRangeSize, 1);
}

Y_UNIT_TEST(AsyncRunOneTaskAndWaitExceptionsOneExtraThread) {
    AsyncRunRangeAndWaitExceptions(1, 1);
}

Y_UNIT_TEST(AsyncRunTwoRangesAndWaitExceptions) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(DefaultThreadsCount);
    TAtomic signal = 0;
    TVector<int> data1(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures1 = localExecutor.ExecRangeWithFutures([&signal, &data1](int i) {
        UNIT_ASSERT(data1[i] == 0);
        while (AtomicGet(signal) == 0)
            ;
        data1[i] += 1;
        throw 15000 + i;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::LOW_PRIORITY);
    TVector<int> data2(DefaultRangeSize, 0);
    TVector<NThreading::TFuture<void>> futures2 = localExecutor.ExecRangeWithFutures([&signal, &data2](int i) {
        UNIT_ASSERT(data2[i] == 0);
        while (AtomicGet(signal) == 0)
            ;
        data2[i] += 2;
        throw 16000 + i;
    },
                                                                                     0, DefaultRangeSize, TLocalExecutor::HIGH_PRIORITY);

    UNIT_ASSERT(AllOf(data1, 0));
    UNIT_ASSERT(AllOf(data2, 0));
    UNIT_ASSERT(futures1.size() == DefaultRangeSize);
    UNIT_ASSERT(futures2.size() == DefaultRangeSize);
    AtomicSet(signal, 1);
    int exceptionsCaught = 0;
    for (int i = 0; i < DefaultRangeSize; ++i) {
        try {
            futures1[i].GetValueSync();
        } catch (int& e) {
            if (e == 15000 + i) {
                ++exceptionsCaught;
            }
        }
        try {
            futures2[i].GetValueSync();
        } catch (int& e) {
            if (e == 16000 + i) {
                ++exceptionsCaught;
            }
        }
    }
    UNIT_ASSERT(exceptionsCaught == 2 * DefaultRangeSize);
    UNIT_ASSERT(AllOf(data1, 1));
    UNIT_ASSERT(AllOf(data2, 2));
}

void RunRangeAndCheckExceptionsWithWaitComplete(int rangeSize, int threadsCount) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadsCount);
    TVector<int> data(rangeSize, 0);
    TVector<NThreading::TFuture<void>> futures = localExecutor.ExecRangeWithFutures([&data](int i) {
        UNIT_ASSERT(data[i] == 0);
        data[i] += 1;
        throw 30000 + i;
    },
                                                                                    0, rangeSize, TLocalExecutor::EFlags::WAIT_COMPLETE);
    UNIT_ASSERT(AllOf(data, 1));
    int exceptionsCaught = 0;
    for (int i = 0; i < rangeSize; ++i) {
        try {
            futures[i].GetValueSync();
        } catch (int& e) {
            if (e == 30000 + i) {
                ++exceptionsCaught;
            }
        }
    }
    UNIT_ASSERT(exceptionsCaught == rangeSize);
    UNIT_ASSERT(AllOf(data, 1));
}

Y_UNIT_TEST(RunRangeAndCheckExceptionsWithWaitComplete) {
    RunRangeAndCheckExceptionsWithWaitComplete(DefaultRangeSize, DefaultThreadsCount);
}

Y_UNIT_TEST(RunOneAndCheckExceptionsWithWaitComplete) {
    RunRangeAndCheckExceptionsWithWaitComplete(1, DefaultThreadsCount);
}

Y_UNIT_TEST(RunRangeAndCheckExceptionsWithWaitCompleteOneExtraThread) {
    RunRangeAndCheckExceptionsWithWaitComplete(DefaultRangeSize, 1);
}

Y_UNIT_TEST(RunOneAndCheckExceptionsWithWaitCompleteOneExtraThread) {
    RunRangeAndCheckExceptionsWithWaitComplete(1, 1);
}

Y_UNIT_TEST(RunRangeAndCheckExceptionsWithWaitCompleteZeroExtraThreads) {
    RunRangeAndCheckExceptionsWithWaitComplete(DefaultRangeSize, 0);
}

Y_UNIT_TEST(RunOneAndCheckExceptionsWithWaitCompleteZeroExtraThreads) {
    RunRangeAndCheckExceptionsWithWaitComplete(1, 0);
}
}
;

Y_UNIT_TEST_SUITE(ExecRangeWithThrow){
    void RunParallelWhichThrowsTTestException(int rangeStart, int rangeSize, int threadsCount, int flags, TAtomic& processed){
        AtomicSet(processed, 0);
TLocalExecutor localExecutor;
localExecutor.RunAdditionalThreads(threadsCount);
localExecutor.ExecRangeWithThrow([&processed](int) {
    AtomicAdd(processed, 1);
    throw TTestException();
},
                                 rangeStart, rangeStart + rangeSize, flags);
}

Y_UNIT_TEST(RunParallelWhichThrowsTTestException) {
    TAtomic processed = 0;
    UNIT_ASSERT_EXCEPTION(
        RunParallelWhichThrowsTTestException(10, 40, DefaultThreadsCount,
                                             TLocalExecutor::EFlags::WAIT_COMPLETE, processed),
        TTestException);
    UNIT_ASSERT(AtomicGet(processed) == 40);
}

void ThrowAndCatchTTestException(int rangeSize, int threadsCount, int flags) {
    TAtomic processed = 0;
    UNIT_ASSERT_EXCEPTION(
        RunParallelWhichThrowsTTestException(0, rangeSize, threadsCount, flags, processed),
        TTestException);
    UNIT_ASSERT(AtomicGet(processed) == rangeSize);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionLowPriority) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE | TLocalExecutor::EFlags::LOW_PRIORITY);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionMedPriority) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE | TLocalExecutor::EFlags::MED_PRIORITY);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionHighPriority) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE | TLocalExecutor::EFlags::HIGH_PRIORITY);
}

Y_UNIT_TEST(ThrowAndCatchTTestExceptionWaitComplete) {
    ThrowAndCatchTTestException(DefaultRangeSize, DefaultThreadsCount,
                                TLocalExecutor::EFlags::WAIT_COMPLETE);
}

Y_UNIT_TEST(RethrowExeptionSequentialWaitComplete) {
    ThrowAndCatchTTestException(DefaultRangeSize, 0,
                                TLocalExecutor::EFlags::WAIT_COMPLETE);
}

Y_UNIT_TEST(RethrowExeptionOneExtraThreadWaitComplete) {
    ThrowAndCatchTTestException(DefaultRangeSize, 1,
                                TLocalExecutor::EFlags::WAIT_COMPLETE);
}

void ThrowsTTestExceptionFromNested(TLocalExecutor& localExecutor) {
    localExecutor.ExecRangeWithThrow([](int) {
        throw TTestException();
    },
                                     0, 10, TLocalExecutor::EFlags::WAIT_COMPLETE);
}

void CatchTTestExceptionFromNested(TAtomic& processed1, TAtomic& processed2) {
    TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(DefaultThreadsCount);
    localExecutor.ExecRangeWithThrow([&processed1, &processed2, &localExecutor](int) {
        AtomicAdd(processed1, 1);
        UNIT_ASSERT_EXCEPTION(
            ThrowsTTestExceptionFromNested(localExecutor),
            TTestException);
        AtomicAdd(processed2, 1);
    },
                                     0, DefaultRangeSize, TLocalExecutor::EFlags::WAIT_COMPLETE);
}

Y_UNIT_TEST(NestedParallelExceptionsDoNotLeak) {
    TAtomic processed1 = 0;
    TAtomic processed2 = 0;
    UNIT_ASSERT_NO_EXCEPTION(
        CatchTTestExceptionFromNested(processed1, processed2));
    UNIT_ASSERT_EQUAL(AtomicGet(processed1), DefaultRangeSize);
    UNIT_ASSERT_EQUAL(AtomicGet(processed2), DefaultRangeSize);
}
}
;

static int GetPriority(size_t i) {
    static const int priorities[] = {
        NPar::TLocalExecutor::HIGH_PRIORITY,
        NPar::TLocalExecutor::MED_PRIORITY,
        NPar::TLocalExecutor::LOW_PRIORITY
    };

    return priorities[i % 3];
}

static TAutoPtr<IThreadPool::IThread> AddUnrelatedJobs(NPar::TLocalExecutor& e) {
    return SystemThreadPool()->Run([&e]{
        e.Exec([](int) {
            // do nothing
        }, 0, NPar::TLocalExecutor::HIGH_PRIORITY);
    });
}

Y_UNIT_TEST_SUITE(SyncronizationTests) {
    Y_UNIT_TEST(TestSyncWithoutUnderlyingThreads) {
        NPar::TLocalExecutor e;
        UNIT_ASSERT_EXCEPTION(e.WaitJobsFromCurrentThread(), std::exception);
    }

    Y_UNIT_TEST(TestOneTaskAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        e.RunAdditionalThreads(3);

        bool flag = false;
        e.Exec([&flag](int){ flag = true; }, 0, NPar::TLocalExecutor::HIGH_PRIORITY);

        e.WaitJobsFromCurrentThread();

        UNIT_ASSERT(flag);
    }

    Y_UNIT_TEST(TestOneTaskAddThreadsAfterExec) {
        NPar::TLocalExecutor e;

        bool flag = false;
        e.Exec([&flag](int){ flag = true; }, 0, NPar::TLocalExecutor::HIGH_PRIORITY);

        e.RunAdditionalThreads(3);
        e.WaitJobsFromCurrentThread();

        UNIT_ASSERT(flag);
    }

    Y_UNIT_TEST(TestMultipleTasksAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        e.RunAdditionalThreads(3);

        TVector<bool> flags(10);
        for (auto& flag : flags) {
            e.Exec([&flag](int){ flag = true; }, 0, NPar::TLocalExecutor::HIGH_PRIORITY);
        }

        e.WaitJobsFromCurrentThread();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestMultipleTasksAddThreadsAfterExec) {
        NPar::TLocalExecutor e;

        TVector<bool> flags(10);
        for (auto& flag : flags) {
            e.Exec([&flag](int){ flag = true; }, 0, NPar::TLocalExecutor::HIGH_PRIORITY);
        }

        e.RunAdditionalThreads(3);
        e.WaitJobsFromCurrentThread();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestMultipleTasksWithDifferentPrioritiesAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        e.RunAdditionalThreads(3);

        TVector<bool> flags(10);
        for (size_t i = 0; i < flags.size(); ++i) {
            e.Exec([flag = &flags[i]](int){ *flag = true; }, 0, GetPriority(i));
        }

        e.WaitJobsFromCurrentThread();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestMultipleTasksWithDifferentPrioritiesAddThreadsAfterExec) {
        NPar::TLocalExecutor e;

        TVector<bool> flags(10);
        for (size_t i = 0; i < flags.size(); ++i) {
            e.Exec([flag = &flags[i]](int){ *flag = true; }, 0, GetPriority(i));
        }

        e.RunAdditionalThreads(3);
        e.WaitJobsFromCurrentThread();

        for (const auto flag : flags) {
            UNIT_ASSERT(flag);
        }
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsBeforeExec) {
        NPar::TLocalExecutor e;
        e.RunAdditionalThreads(3);

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec([flag = &flags[i]](int){ *flag = true; }, 0, GetPriority(i));
                }
            }));
        }

        for (const auto& thread : threads) {
            thread->Join();
        }

        e.WaitJobsFromCurrentThread();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsAfterExec) {
        NPar::TLocalExecutor e;

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec([flag = &flags[i]](int){ *flag = true; }, 0, GetPriority(i));
                }
            }));
        }

        e.RunAdditionalThreads(3);

        for (const auto& thread : threads) {
            thread->Join();
        }

        e.WaitJobsFromCurrentThread();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsBeforeExecAndUnrelatedJobs) {
        NPar::TLocalExecutor e;
        e.RunAdditionalThreads(3);

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec([flag = &flags[i]](int){ *flag = true; }, 0, GetPriority(i));
                }
            }));
        }

        // these jobs may be added before or after `Sync` invocation, it's for thread sanitizer to
        // increase number of possible cases
        const auto unrelated = AddUnrelatedJobs(e);

        for (const auto& thread : threads) {
            thread->Join();
        }

        e.WaitJobsFromCurrentThread();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }

        unrelated->Join();
    }

    Y_UNIT_TEST(TestFromMultipleThreadsMultipleTasksWithDifferentPrioritiesAddThreadsAfterExecAndUnrelatedJobs) {
        NPar::TLocalExecutor e;

        TVector<TAutoPtr<IThreadPool::IThread>> threads;
        TVector<bool> flagsArr[2];
        flagsArr[0].resize(10);
        flagsArr[1].resize(10);
        for (auto& flags : flagsArr) {
            threads.push_back(SystemThreadPool()->Run([&e, &flags]{
                for (size_t i = 0; i < flags.size(); ++i) {
                    e.Exec([flag = &flags[i]](int){ *flag = true; }, 0, GetPriority(i));
                }
            }));
        }

        // these jobs may be added before or after `Sync` invocation, it's for thread sanitizer to
        // increase number of possible cases
        const auto unrelated = AddUnrelatedJobs(e);

        e.RunAdditionalThreads(3);

        for (const auto& thread : threads) {
            thread->Join();
        }

        e.WaitJobsFromCurrentThread();

        for (const auto& flags : flagsArr) {
            for (const auto flag : flags) {
                UNIT_ASSERT(flag);
            }
        }

        unrelated->Join();
    }
}

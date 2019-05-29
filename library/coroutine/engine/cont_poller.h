#pragma once

#include "poller.h"
#include "sockmap.h"

#include <library/containers/intrusive_rb_tree/rb_tree.h>

#include <util/datetime/base.h>
#include <util/memory/pool.h>
#include <util/memory/smallobj.h>
#include <util/network/init.h>

#include <cerrno>


class TCont;
class TContExecutor;
class TFdEvent;

namespace NCoro {

    class IPollEvent;

    void Reshedule(TCont* cont) noexcept;

    void RemoveFromPoller(TCont* cont, IPollEvent* event) noexcept;


    struct TContPollEventCompare {
        template <class T>
        static inline bool Compare(const T& l, const T& r) noexcept {
            return l.DeadLine() < r.DeadLine() || (l.DeadLine() == r.DeadLine() && &l < &r);
        }
    };


    class TContPollEvent : public TRbTreeItem<TContPollEvent, TContPollEventCompare> {
    public:
        TContPollEvent(TCont* cont, TInstant deadLine) noexcept
            : Cont_(cont)
            , DeadLine_(deadLine)
            , Status_(EINPROGRESS)
        {}

        static bool Compare(const TContPollEvent& l, const TContPollEvent& r) noexcept {
            return l.DeadLine() < r.DeadLine() || (l.DeadLine() == r.DeadLine() && &l < &r);
        }

        int Status() const noexcept {
            return Status_;
        }

        void SetStatus(int status) noexcept {
            Status_ = status;
        }

        TCont* Cont() noexcept {
            return Cont_;
        }

        TInstant DeadLine() const noexcept {
            return DeadLine_;
        }

        void Wake(int status) noexcept {
            SetStatus(status);
            Wake();
        }

    private:
        void Wake() noexcept {
            UnLink();
            Reshedule(Cont_);
        }

    private:
        TCont* Cont_;
        TInstant DeadLine_;
        int Status_;
    };


    class IPollEvent: public TIntrusiveListItem<IPollEvent> {
    public:
        IPollEvent(SOCKET fd, ui16 what) noexcept
            : Fd_(fd)
            , What_(what)
        {}

        virtual ~IPollEvent() {}

        SOCKET Fd() const noexcept {
            return Fd_;
        }

        int What() const noexcept {
            return What_;
        }

        virtual void OnPollEvent(int status) noexcept = 0;

    private:
        SOCKET Fd_;
        ui16 What_;
    };


    template <class T>
    class TBigArray {
        struct TValue: public T, public TObjectFromPool<TValue> {
            TValue() {}
        };

    public:
        TBigArray()
            : Pool_(TMemoryPool::TExpGrow::Instance(), TDefaultAllocator::Instance())
        {}

        T* Get(size_t index) {
            TRef& ret = Lst_.Get(index);
            if (!ret) {
                ret = new (&Pool_) TValue();
            }
            return ret.Get();
        }

    private:
        using TRef = THolder<TValue>;
        typename TValue::TPool Pool_;
        TSocketMap<TRef> Lst_;
    };


    using TPollEventList = TIntrusiveList<IPollEvent>;

    class TContPoller {
    public:
        typedef IPollerFace::TEvent TEvent;
        typedef IPollerFace::TEvents TEvents;

        TContPoller()
            : P_(IPollerFace::Default())
        {
        }

        explicit TContPoller(THolder<IPollerFace> poller)
            : P_(std::move(poller))
        {}

        void Schedule(IPollEvent* event) {
            auto* lst = Lists_.Get(event->Fd());
            const ui16 oldFlags = Flags(*lst);
            lst->PushFront(event);
            const ui16 newFlags = Flags(*lst);

            if (newFlags != oldFlags) {
                P_->Set(lst, event->Fd(), newFlags);
            }
        }

        void Remove(IPollEvent* event) noexcept {
            auto* lst = Lists_.Get(event->Fd());
            const ui16 oldFlags = Flags(*lst);
            event->Unlink();
            const ui16 newFlags = Flags(*lst);

            if (newFlags != oldFlags) {
                P_->Set(lst, event->Fd(), newFlags);
            }
        }

        void Wait(TEvents& events, TInstant deadLine) {
            events.clear();
            P_->Wait(events, deadLine);
        }

    private:
        static ui16 Flags(TIntrusiveList<IPollEvent>& lst)  noexcept {
            ui16 ret = 0;
            for (auto&& item : lst) {
                ret |= item.What();
            }
            return ret;
        }

    private:
        TBigArray<TPollEventList> Lists_;
        THolder<IPollerFace> P_;
    };


    class TContPollEventHolder {
    public:
        TContPollEventHolder(void* memory, TCont* rep, SOCKET fds[], int what[], size_t nfds, TInstant deadline);
        ~TContPollEventHolder();

        void ScheduleIoWait(TContExecutor* executor) noexcept;

        TFdEvent* TriggeredEvent() noexcept;

    private:
        TFdEvent* Events_;
        size_t Count_;
    };
}

class TFdEvent final:
    public NCoro::TContPollEvent,
    public NCoro::IPollEvent
{
public:
    TFdEvent(TCont* cont, SOCKET fd, ui16 what, TInstant deadLine) noexcept
        : TContPollEvent(cont, deadLine)
        , IPollEvent(fd, what)
    {}

    ~TFdEvent() {
        RemoveFromIOWait();
    }

    void RemoveFromIOWait() noexcept {
        RemoveFromPoller(Cont(), this);
    }

    void OnPollEvent(int status) noexcept override {
        Wake(status);
    }
};


class TTimerEvent: public NCoro::TContPollEvent {
public:
    TTimerEvent(TCont* cont, TInstant deadLine) noexcept
        : TContPollEvent(cont, deadLine)
    {}
};
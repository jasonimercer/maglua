#include "jthread.h"

static void* thread_call_wrapper(void* arg)
{
	JThread* jt = (JThread*)arg;
	
	return jt->call_from_wrapper();
}

JThread::JThread(void *(*_start_routine)(void*), void* _arg)
{
	state = unstarted;
	start_routine = _start_routine;
	arg = _arg;
}

JThread::~JThread()
{
	
}


int JThread::start()
{
	if(state == running)
		return 1;
	
	pthread_create(&thread, 0, thread_call_wrapper, this);
	return 0;
}

void* JThread::call_from_wrapper()
{
	void* v = start_routine(arg);
	state = complete;
	return v;
}

int JThread::join()
{
	pthread_join(thread, 0);
}

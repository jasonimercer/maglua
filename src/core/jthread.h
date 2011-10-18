#ifndef WIN32

#ifndef JASONTHREAD
#define JASONTHREAD

#include <pthread.h>

class JThread
{
public:
	JThread(void *(*start_routine)(void*), void* arg);
	~JThread();
	int start();

	enum jstate
	{
		unstarted,
		running,
		complete
	};

	jstate state;
	
	void* call_from_wrapper();
	int join();
private:
	void* (*start_routine)(void*);
	void* arg;
	
	pthread_t thread;
};

#endif

#endif

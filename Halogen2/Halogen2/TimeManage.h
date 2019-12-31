#pragma once
#include <time.h>
#include <atomic>
#include <mutex>

extern std::atomic<bool> KeepSearching;

class Timer
{
public:
	Timer();
	~Timer();

	void Start();
	void Restart();
	int ElapsedMs();

private:
	double ElapsedTime;

	clock_t Begin;
	clock_t End;
};

class SearchTimeManage
{
public:
	SearchTimeManage();
	~SearchTimeManage();

	bool ContinueSearch();	//Should I search to another depth, or stop with what ive got?
	bool AbortSearch();		//should I attempt to stop searching right now?

	void StartSearch(int ms);	//pass the allowed search time maximum in milliseconds

private:
	Timer timer;
	int AllowedSearchTimeMS;
};


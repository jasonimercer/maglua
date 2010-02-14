#ifndef SPINOPERATIONEXCHANGE
#define SPINOPERATIONEXCHANGE

#include "spinoperation.h"

class Exchange : public SpinOperation
{
public:
	Exchange(int nx, int ny, int nz);
	virtual ~Exchange();
	
	bool apply(SpinSystem* ss);

	void addPath(int site1, int site2, double strength);

	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);
	
private:
	int size;
	int num;
	
	int* fromsite;
	int* tosite;
	double* strength;
};

Exchange* checkExchange(lua_State* L, int idx);
void registerExchange(lua_State* L);


#endif

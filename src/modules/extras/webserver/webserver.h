/******************************************************************************
* Copyright (C) 2008-2015 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/


#ifndef _WEBSERVER_DEF
#define _WEBSERVER_DEF

#include <semaphore.h>
#include "luabaseobject.h"

class WebServer : public LuaBaseObject
{
public:
    WebServer();
    ~WebServer();

    LINEAGE1("WebServer");
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L, int base=1);
    static int help(lua_State* L);

    void encode(buffer* b);
    int decode(buffer* b);


    int main(lua_State* L);

    int handleClientMessage(const char* msg, int session_id, int sock_fd);

    void init();
    void deinit();

    int getInternalData(lua_State* L);
    void setInternalData(lua_State* L, int stack_pos);


    int port;
    sem_t mutex;

    void lock();
    void unlock();

    int data_ref;
    int ref_client_function;
};

#endif

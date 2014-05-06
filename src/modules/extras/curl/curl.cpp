#include <luabaseobject.h>
#include "curl_luafuncs.h"
#include <curl/curl.h>
#include <string.h>
#include <stdlib.h>

#include "info.h"
extern "C"
{
int lib_register(lua_State* L);
int lib_version(lua_State* L);
const char* lib_name(lua_State* L);
int lib_main(lua_State* L);
}


static size_t accumulate(void *buffer, size_t size, size_t nmemb, void* arg)
{
	char** dest = (char**) arg;
	int end = strlen(*dest);

	*dest = (char*)realloc(*dest, end + size * nmemb + 1);

	memcpy((*dest) + end, buffer, size*nmemb);

	(*dest)[end + size * nmemb] = 0;

	return nmemb;
}

static int l_curl_fetch(lua_State* L)
{
	const char* resource = lua_tostring(L, 1);
	CURL *curl;
	CURLcode res;
	std::string err = "";

	char* dest = (char*)malloc(4);
	dest[0] = 0;

	curl = curl_easy_init();
	if(curl) 
	{
		curl_easy_setopt(curl, CURLOPT_URL, resource);

		/* could be redirected, so we tell libcurl to follow redirection */ 
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
 
		/* Define our callback to get called when there's data to be written */ 
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, accumulate);

		/* Set a pointer to our struct to pass to the callback */ 
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &dest);

		/* Perform the request, res will get the return code */ 
		res = curl_easy_perform(curl);

		/* Check for errors */ 
		if(res != CURLE_OK)
			err = curl_easy_strerror(res);
 
		/* always cleanup */ 
		curl_easy_cleanup(curl);
	}

	if(res == CURLE_OK)
	{
		lua_pushstring(L, dest);
		free(dest);
		return 1;
	}

	free(dest);

	lua_pushboolean(L, 0);
	lua_pushstring(L, err.c_str());
	return 2;
}


int lib_register(lua_State* L)
{
	lua_pushcfunction(L, l_curl_fetch);
	lua_setglobal(L, "curl");

	const char* s = __curl_luafuncs();

	if(luaL_dostringn(L, s, "curl_luafuncs.lua"))
	{
		fprintf(stderr, "CURL: %s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}
	
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Curl";
#else
	return "Curl-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}

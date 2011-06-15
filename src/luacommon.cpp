// /******************************************************************************
// * Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
// *
// * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ******************************************************************************/
// 
// #include <iostream>
// #include <string>
// #include <vector>
// #include <string.h>
// #include <errno.h>
// 
// using namespace std;
// 
// #include <dlfcn.h>
// #include <dirent.h>
// #include "main.h"
// #include "luacommon.h"
// 
// #include "spinsystem.h"
// #include "luacommon.h"
// #include "spinoperation.h"
// #include "llg.h"
// #include "llgquat.h"
// #include "spinoperationexchange.h"
// #include "spinoperationappliedfield.h"
// #include "spinoperationanisotropy.h"
// #include "spinoperationdipole.h"
// #include "spinoperationdipoledisordered.h"
// #include "mersennetwister.h"
// #include "spinoperationthermal.h"
// #include "interpolatingfunction.h"
// #include "interpolatingfunction2d.h"
// #include "luampi.h"
// 	
// 	registerSpinSystem(L);
// 	registerLLG(L);
// 	registerExchange(L);
// 	registerAppliedField(L);
// 	registerAnisotropy(L);
// 	registerDipole(L);
// 	registerRandom(L);
// 	registerThermal(L);
// 	registerInterpolatingFunction(L);
// 	registerInterpolatingFunction2D(L);
// 	registerDipoleDisordered(L);
// // 	registerMagnetostatic(L);
// 	
// 	//registerSQLite(L);
// 	
//     
//     // close the library
// //     cout << "Closing library...\n";
// //     dlclose(handle);
// 	
// 	
// //	registerLuaClient(L);
// #ifdef _MPI
// 	registerMPI(L);
// #endif
// }

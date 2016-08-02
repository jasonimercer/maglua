# Copyright (C) 2012 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

OBJECTS=CGS.lo PathDynamics.lo ColorMap.lo RungeKutta.lo POVRay.lo PredictorCorrector.lo AdaptiveTimeStep.lo MakeMacro.lo scripts.o
LIBNAME=Scripts
CATEGORY=common
DEPEND=luabaseobject

BASEDIR=../../..
include $(BASEDIR)/makefile.common.cpu
include $(BASEDIR)/makefile.common.mpi

EXTRA_CLEAN=ColorMap.h ColorMap.cpp CGS.h RungeKutta.h POVRay.h CGS.cpp RungeKutta.cpp POVRay.cpp PredictorCorrector.h PredictorCorrector.cpp AdaptiveTimeStep.h AdaptiveTimeStep.cpp PathDynamics.cpp PathDynamics.h

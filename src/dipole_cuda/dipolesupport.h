/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

void dipoleLoad(
	const int nx, const int ny, const int nz,
	const int gmax, double* ABC,
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ);

double gamma_xx_dip(double rx, double ry, double rz);
double gamma_xy_dip(double rx, double ry, double rz);
double gamma_xz_dip(double rx, double ry, double rz);
double gamma_yy_dip(double rx, double ry, double rz);
double gamma_yz_dip(double rx, double ry, double rz);
double gamma_zz_dip(double rx, double ry, double rz);

	
#include "array_core_matrix_cpu.h"
#include "array_core_matrix_cpu_lapack.h"

template <typename T>
static void matminor(const T* A, const int nx, const int ny, int i, int j, T* dest)
{
	for(int y=0; (y<j) && (y<ny); y++)
	{
		for(int x=0; (x<i) && (x<nx); x++)
		{
			dest[x + (y*(nx-1))] = A[x+nx*y];
		}
		for(int x=i+1; x<nx; x++)
		{
			// printf("A[%i,%i] -> m[%i,%i]\n", x-1,y,x,y);
			dest[(x-1) + (y*(nx-1))] = A[x+nx*y];
		}
	}

	for(int y=j+1; y<ny; y++)
	{
		for(int x=0; x<i && x<nx; x++)
		{
			// printf("A[%i,%i] -> m[%i,%i]\n", x,y-1,x,y);
			dest[x + ((y-1)*(nx-1))] = A[x+nx*y];
		}
		for(int x=i+1; x<nx; x++)
		{
			// printf("A[%i,%i] -> m[%i,%i]\n", x-1,y-1,x,y);
			dest[(x-1) + ((y-1)*(nx-1))] = A[x+nx*y];
		}
	}
}


template <typename T>
static T matdet(const T* A, const int nx, const int ny, bool& ok)
{
	ok = true;
	if(nx != ny)
	{
		ok = false;
		return A[0];
	}

	if(nx == 0)
	{
		return 0;
	}

	if(nx == 1)
	{
		return A[0];
	}

	if(nx == 2)
	{
		return A[0]*A[3] - A[1]*A[2];
	}

	T* m = new T[(nx-1)*(ny-1)];
	T sum = 0;
	for(int i=0; i<nx; i++)
	{
#if 0
		cout << "making minor " << i << ", 0 for matrix:" << endl;
		for(int a=0; a<nx; a++)
		{
			for(int b=0; b<nx; b++)
			{
				cout << A[b*nx + a] << "\t";
			}
			cout << endl;
		}
#endif
		matminor(A, nx, ny, i, 0, m);
#if 0
		cout << "minor:" << endl;
		for(int a=0; a<nx-1; a++)
		{
			for(int b=0; b<nx-1; b++)
			{
				cout << m[b*(nx-1) + a] << "\t";
			}
			cout << endl;
		}
#endif

		if(i & 0x1) //odd, negative
		{
			sum = sum - A[i] * matdet(m, nx-1, ny-1, ok);
		}
		else  //even, positive
		{
			sum = sum + A[i] * matdet(m, nx-1, ny-1, ok);
		}
	}
	delete [] m;

	return sum;
}


template <typename T>
bool mattrans(const T* src, const int* AB, const int* dims_in, T* dest, const int* dims_out)
{
	if(AB[0] == AB[1])
	{
		if(src == dest)
			return true;
		
		if(src != dest )
			memcpy(dest, src, sizeof(T)*dims_in[0]*dims_in[1]*dims_in[2]);
	}

	int nx = dims_in[0];
	int ny = dims_in[1];
	int nz = dims_in[2];
	
	int dx = dims_out[0];
	int dy = dims_out[1];
	int dz = dims_out[2];

	if(AB[0] == 0 && AB[1] == 1) // XY
	{
		if(src == dest)
		{
			for(int k=0; k<nz; k++)
				for(int j=0; j<ny; j++)
					for(int i=0; i<j; i++)
					{
						const T t = dest[i + j*nx + k*nx*ny];
						dest[i + j*nx + k*nx*ny] = dest[j + nx*i + nx*ny*k];
						dest[j + nx*i + nx*ny*k] = t;
					}
		}
		else
		{
			for(int k=0; k<nz; k++)
				for(int j=0; j<ny; j++)
					for(int i=0; i<nx; i++)
					{
						dest[j + i*dx + dy*dx*k] = src[i + j*nx + k*nx*ny];
					}
		}		
	}
	
	if(AB[0] == 0 && AB[1] == 2) //XZ
	{
		if(src == dest)
		{
			for(int k=0; k<nz; k++)
				for(int j=0; j<ny; j++)
					for(int i=0; i<k; i++)
					{
						const T t = dest[i + j*nx + k*nx*ny];
						dest[i + j*nx + k*nx*ny] = dest[k + nx*j + nx*ny*i];
						dest[k + nx*j + nx*ny*i] = t;
					}
		}
		else
		{
			for(int k=0; k<nz; k++)
				for(int j=0; j<ny; j++)
					for(int i=0; i<nx; i++)
					{
						dest[k + j*dx + i*dx*dy] = src[i + j*nx + k*nx*ny];
					}
		}		
	}
	
		
	if(AB[0] == 1 && AB[1] == 2) //YZ
	{
		if(src == dest)
		{
			for(int k=0; k<nz; k++)
				for(int j=0; j<k; j++)
					for(int i=0; i<nx; i++)
					{
						const T t = dest[i + nx*j + nx*ny*k];
						dest[i + nx*j + nx*ny*k] = dest[i + nx*k + nx*ny*j];
						dest[i + nx*k + nx*ny*j] = t;
					}
		}
		else
		{
			for(int k=0; k<nz; k++)
				for(int j=0; j<ny; j++)
					for(int i=0; i<nx; i++)
					{
						dest[i + k*dx + j*dx*dy] = src[i + j*nx + k*nx*ny];
					}
		}		
	}
	
	return true;
}

static int swap_components(int* src, int c1, int c2, int* dest)
{
	memcpy(dest, src, sizeof(int)*3);
	dest[c1] = src[c2];
	dest[c2] = src[c1];
}

template <typename T>
static int l_mattrans(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);
	
	char c[3] = "XY";
		
	Array<T>* B = 0;
	if(luaT_is< Array<T> >(L, 2))
		B = luaT_to< Array<T> >(L, 2);

	for(int i=2; i<=lua_gettop(L); i++)
	{
		if(lua_isstring(L, i))
		{
			const char* ab = lua_tostring(L, i);
			for(int j=0; (j<2) && (ab[j] != 0); j++)
			{
				c[j] = ab[j];
			}
		}
	}

	int AB[2] = {0,1};
	for(int i=0; i<2; i++)
	{
		if(c[i] == 'x' || c[i] == 'X') AB[i] = 0;
		if(c[i] == 'y' || c[i] == 'Y') AB[i] = 1;
		if(c[i] == 'z' || c[i] == 'Z') AB[i] = 2;
	}
	
	if(AB[0] > AB[1]) //lowest first
	{
		int t = AB[0];
		AB[0] = AB[1];
		AB[1] = t;
	}
	
	int dims_in[3];
	int dims_out[3];
	dims_in[0] = A->nx;
	dims_in[1] = A->ny;
	dims_in[2] = A->nz;

	swap_components(dims_in, AB[0], AB[1], dims_out);
	
	if(B)
	{
		if( dims_out[0] != B->nx ||
			dims_out[1] != B->ny ||
			dims_out[2] != B->nz)
		{
			return luaL_error(L, "Destination Array size mismatch");
		}
	}
	else
	{
		B = new Array<T>(dims_out[0], dims_out[1], dims_out[2]);
	}

	mattrans(A->data(), AB, dims_in, B->data(), dims_out);

	luaT_push< Array<T> >(L, B);
	return 1;
}




#if 0
template <typename T>
static void mm(
	const T* A, const int ra, const int ca,
	const T* B, const int rb, const int cb,
	      T* C, const int rc, const int cc)
{
	for(int r=0; r<ra; r++)
	{
		for(int c=0; c<cb; c++)
		{
			T sum = 0;
			for(int k=0; k<ca; k++)
			{
				sum += A[r*ca + k] * B[k*cb + c];
			}
			C[r*cc + c] = sum;
		}
	}
}
#endif
		
#if 0
template <typename T>
static int lT_matmul(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);
	LUA_PREAMBLE(Array<T>, B, 2);
	
	if(A->nx != B->ny)
		return luaL_error(L, "Column count of A (%d) does not match row count of B (%d)", A->nx, B->ny);
	
	Array<T>* C = 0;
	if(luaT_is<Array<T> >(L, 3))
		C = luaT_to< Array<T> >(L, 3);
	else
		C = new Array<T>(B->nx, A->ny);
	
	if(C->nx != B->nx || C->ny != A->ny)
		return luaL_error(L, "Size mismatch for destination matrix");
	
	mm<T>(A->data(), A->ny, A->nx, B->data(), B->ny, B->nx, C->data(), C->ny, C->nx);
	
	luaT_push< Array<T> >(L, C);
	return 1;
}

static int l_matmul(lua_State* L)
{
	if(luaT_is<dArray>(L, 1))
		return lT_matmul<double>(L);
	if(luaT_is<fArray>(L, 1))
		return lT_matmul<float>(L);
	return luaL_error(L, "Array.matMul is only implemented for single and double precision arrays");
}
#endif


template <typename T>
static int lT_matmakei(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);

	A->zero();
	int m = A->nx;
	if(A->ny < m)
		m = A->ny;
	
	T* d = A->data();
	
	for(int i=0; i<m; i++)
	{
		d[i*A->nx + i] = luaT<T>::one();
	}
	luaT_push<Array<T> >(L, A);
	return 1;	
}
static int l_matmakei(lua_State* L)
{
	if(luaT_is<dArray>(L, 1))
		return lT_matmakei<double>(L);
	if(luaT_is<fArray>(L, 1))
		return lT_matmakei<float>(L);
	return luaL_error(L, "Array.matMakeI is only implemented for single and double precision arrays");
}





template <typename T>
static int lT_matdet(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);
	
	if(A->nx != A->ny)
		return luaL_error(L, "Matrix is not square");
	
	bool ok;
	
	T res = matdet<T>(A->data(), A->nx, A->ny, ok);
	
	luaT<T>::push(L, res);
    return luaT<T>::elements();
}

static int l_matdet(lua_State* L)
{
	if(luaT_is<dArray>(L, 1))
		return lT_matdet<double>(L);
	if(luaT_is<fArray>(L, 1))
		return lT_matdet<float>(L);
	if(luaT_is<iArray>(L, 1))
		return lT_matdet<int>(L);
	return luaL_error(L, "Array.matDet is only implemented for single and double precision and integer arrays");
}






template<typename T, int type>
static int lT_mat_tri(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	Array<T>* b = 0;
	bool diag = false;
	for(int i=2; i<=lua_gettop(L); i++)
	{
		if(luaT_is< Array<T> >(L, i))
			b = luaT_to< Array<T> >(L, i);
		if(lua_isboolean(L, i))
			diag |= lua_toboolean(L, i);
	}

	if(!b)
	{
		b = new Array<T>(a->nx, a->ny, a->nz);
	}
	b->copyFrom(a);
	
	if(type > 0) //upper, clear strictly lower
	{
		for(int y=1; y<b->ny; y++)
			for(int x=0; x<y && x<b->nx; x++)
				b->set(x,y,0, luaT<T>::zero());
	}
	if(type < 0) //lower, clear strictly upper
	{
		for(int y=0; y<b->ny; y++)
			for(int x=y+1; x<b->nx; x++)
			{
				b->set(x,y,0, luaT<T>::zero());
			}
	}
	
	if(!diag) //clear the diagonal
	{
		for(int i=0; (i<b->nx) && (i<b->ny); i++)
			b->set(i,i,0, luaT<T>::zero());
	}
	luaT_push< Array<T> >(L, b);
	return 1;
}

static int l_matupper(lua_State* L)
{
	if(luaT_is<dArray>(L, 1)) return lT_mat_tri<double, 1>(L);
	if(luaT_is<fArray>(L, 1)) return lT_mat_tri<float,  1>(L);
	if(luaT_is<iArray>(L, 1)) return lT_mat_tri<int,    1>(L);
	if(luaT_is<dcArray>(L, 1)) return lT_mat_tri<doubleComplex, 1>(L);
	if(luaT_is<fcArray>(L, 1)) return lT_mat_tri<floatComplex, 1>(L);
	return luaL_error(L, "unknown data type");
}

static int l_matlower(lua_State* L)
{
	if(luaT_is<dArray>(L, 1)) return lT_mat_tri<double, -1>(L);
	if(luaT_is<fArray>(L, 1)) return lT_mat_tri<float,  -1>(L);
	if(luaT_is<iArray>(L, 1)) return lT_mat_tri<int,    -1>(L);
	if(luaT_is<dcArray>(L, 1)) return lT_mat_tri<doubleComplex, -1>(L);
	if(luaT_is<fcArray>(L, 1)) return lT_mat_tri<floatComplex, -1>(L);
	return luaL_error(L, "unknown data type");
}




template<typename T>
static const luaL_Reg* get_base_methods_matrix_()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"matTrans",     l_mattrans<T>},
		{"matDet",       l_matdet},
		{"matMakeI",     l_matmakei},
		{"matUpper",     l_matupper},
		{"matLower",     l_matlower},
		
		{"matMul",       l_matmul},     // BLAS DGEMM
		{"matEigen",     l_mateigen},   // LAPACK
		{"matInv",       l_matinverse}, // LAPACK
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

const luaL_Reg* get_base_methods_matrix_int() {return get_base_methods_matrix_<int>();}
const luaL_Reg* get_base_methods_matrix_float() {return get_base_methods_matrix_<float>();}
const luaL_Reg* get_base_methods_matrix_double() {return get_base_methods_matrix_<double>();}
const luaL_Reg* get_base_methods_matrix_floatComplex() {return get_base_methods_matrix_<floatComplex>();}
const luaL_Reg* get_base_methods_matrix_doubleComplex()  {return get_base_methods_matrix_<doubleComplex>();}





















template<typename T>
static int Array_help_matrix_(lua_State* L)
{
	lua_CFunction func = lua_tocfunction(L, 1);

	lua_CFunction f18 = l_mattrans<T>;
	if(func == f18)
	{
		lua_pushstring(L, "Transpose the X and Y components of the array");
		lua_pushstring(L, "1 Optional String, 1 Optional Array: By default the XY components of the array will be swapped. If a string is supplied then those components will be swapped. Expected values: \"XY\", \"Zx\", \"yx\", etc. Destination array which will contain the transpose, may be the calling array. Must be of the appropriate dimensions.");
		lua_pushstring(L, "1 Array: Transpose of array. If the optional array is given it will be the same array otherwaise a new array will be created.");
		return 3;
	}
	
	lua_CFunction f19 = l_matdet;
	if(func == f19)
	{
		lua_pushstring(L, "Treat array like a matrix and compute the determinant of the z=1 slice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Value: The determinant.");
		return 3;
	}
	
// 	lua_CFunction f20 = l_matmul;
// 	if(func == f20)
// 	{
// 		lua_pushstring(L, "Treat arrays like matrices and do Matrix Multiplication on the z=1 layer");
// 		lua_pushstring(L, "1 Array, 1 Optional Array: The given array will be multiply the calling Array, their dimensions must match to allow legal matrix multiplication. If a 2nd Array is supplied the product will be stored in it, otherise a new Array will be created.");
// 		lua_pushstring(L, "1 Array: The product of the multiplication.");
// 		return 3;
// 	}
	
	
	lua_CFunction f21 = l_matmakei;
	if(func == f21)
	{
		lua_pushstring(L, "Make the calling array the Identity matrix of dimensions equal to the array (z=1 only)");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: The same array as the calling array. Useful for chaining.");
		return 3;
	}
	
	if(func == &l_matupper)
	{
		lua_pushstring(L, "Using the provided or new array, copy in the upper triangular part of the matrix.");
		lua_pushstring(L, "1 Optional Array, 1 Optional boolean: If an array is provided the result will be put in it, otherwise a new array will be made. By default the diagonal is not copied into the result unless the boolean value is true.");
		lua_pushstring(L, "1 Array: The upper triangluar matrix.");
		return 3;
	}
		
	if(func == &l_matlower)
	{
		lua_pushstring(L, "Using the provided or new array, copy in the lower triangular part of the matrix.");
		lua_pushstring(L, "1 Optional Array, 1 Optional boolean: If an array is provided the result will be put in it, otherwise a new array will be made. By default the diagonal is not copied into the result unless the boolean value is true.");
		lua_pushstring(L, "1 Array: The lower triangluar matrix.");
		return 3;
	}

	return 0;
}

int Array_help_matrix_int(lua_State* L) {return Array_help_matrix_<int>(L);}
int Array_help_matrix_float(lua_State* L) {return Array_help_matrix_<float>(L);}
int Array_help_matrix_double(lua_State* L) {return Array_help_matrix_<double>(L);}
int Array_help_matrix_floatComplex(lua_State* L) {return Array_help_matrix_<floatComplex>(L);}
int Array_help_matrix_doubleComplex(lua_State* L) {return Array_help_matrix_<doubleComplex>(L);}

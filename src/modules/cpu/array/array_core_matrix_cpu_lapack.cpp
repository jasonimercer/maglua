#include "array_core_matrix_cpu_lapack.h"
#include "array.h"
#include <stdlib.h>


template <typename T>
void __transpose(T* dest, T* src, int nx, int ny)
{
	if(src != dest)
	{
		int c = 0;
		for(int x=0; x<nx; x++)
		{
			for(int y=0; y<ny; y++)
			{
				dest[y*nx+x] = src[c];
				c++;
			}
		}
	}
	else
	{
		// hack until I think of the right way to do things inline
		// square case is easy
		T* A = new T[nx*ny];
		__transpose<T>(A, src, nx,ny);
		memcpy(dest, A, sizeof(T)*nx*ny);
		delete [] A;
	}
}


template <typename T>
void __transpose_square(T* dest, T* src, int n)
{
	__transpose<T>(dest, src, n, n);
}

extern "C"
{
	// eigenvalues/eigenvectors
	int dgeev_(char *jobvl, char *jobvr, int *n, double *a, 
			int *lda, double *wr, double *wi, double *vl, 
			int *ldvl, double *vr, int *ldvr, double *work, 
			int *lwork, int *info);

    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
	
	
	// mat mul
	int dgemm_(char *transa, char *transb, int *m, int *
			n, int *k, double *alpha, double *a, int *lda, 
			double *b, int *ldb, double *beta, double *c, int 
			*ldc);
	int sgemm_(char *transa, char *transb, int *m, int *
			n, int *k, float *alpha, float *a, int *lda, 
			float *b, int *ldb, float *beta, float *c, int 
			*ldc);
}

static int l_mateigen_d(lua_State* L)
{
	LUA_PREAMBLE(Array<double>, A, 1);
	
	if(A->nx != A->ny)
		return luaL_error(L, "matrix must be square");
	int N = A->nx;

	Array<double>* B = 0;
	Array<double>* C = 0;
	
	if(luaT_is<Array<double> >(L, 3))
	{
		B = luaT_to<Array<double> >(L, 3);
		if(B->nx != A->nx || B->ny != A->ny)
			return luaL_error(L, "Destination matrix must be the same size as the calling matrix");
	}
	else
	{
		B = new Array<double>(N,N);
	}

	if(luaT_is<Array<double> >(L, 2))
	{
		C = luaT_to<Array<double> >(L, 2);
		if(C->nx * C->ny != A->nx)
			return luaL_error(L, "Destination vector must be 1xN where N is the row count of the collaing matrix");
	}
	else
	{
		C = new Array<double>(N,1);
	}
	
	
	
	
	double* AT = (double*)malloc(sizeof(double)*N*N);
	__transpose_square<double>(AT, A->data(), N);
	
	char JOBVL ='N';   // Compute Right eigenvectors
	char JOBVR ='V';   // Do not compute Left eigenvectors

	double VL[1];
	int LDVL = 1;
	int LDVR = N;
	int LDA = N;
	
	int LWORK = 4*N; 

	double* WORK =  (double*)malloc(LWORK*sizeof(double));

	int INFO;
	
	double* eigenvectors = B->data();
	double* eigenvalues  = C->data();
	
	double* imag_part = (double*)malloc(sizeof(double)*N);
	
	dgeev_(&JOBVL, &JOBVR, &N, AT, 
				&LDA, eigenvalues, imag_part, 
				VL, &LDVL, 
				eigenvectors, &LDVR,
				WORK, &LWORK, &INFO);
	
	free(AT);
	free(imag_part);
	free(WORK);
	
	
	
	luaT_push< Array<double> >(L, C);
	luaT_push< Array<double> >(L, B);
	
	return 2;
}

int l_mateigen(lua_State* L)
{
	if(luaT_is<dArray>(L, 1)) return l_mateigen_d(L);
// 	if(luaT_is<fArray>(L, 1)) return lT_mat_tri<float,  -1>(L);
// 	if(luaT_is<iArray>(L, 1)) return lT_mat_tri<int,    -1>(L);
// 	if(luaT_is<dcArray>(L, 1)) return lT_mat_tri<doubleComplex, -1>(L);
// 	if(luaT_is<fcArray>(L, 1)) return lT_mat_tri<floatComplex, -1>(L);
	return luaL_error(L, "Unimplemented data type for :matEigen");
}














static int l_matinverse_d(lua_State* L)
{
	LUA_PREAMBLE(Array<double>, A, 1);
	
	if(A->nx != A->ny)
		return luaL_error(L, "matrix must be square");
	int N = A->nx;

	double* AT = new double[N*N];
	
	__transpose_square<double>(AT, A->data(), N);
	
	
	Array<double>* B = 0;
	
	if(luaT_is<Array<double> >(L, 2))
	{
		B = luaT_to<Array<double> >(L, 2);
		if(B->nx != A->nx || B->ny != A->ny)
			return luaL_error(L, "Destination matrix must be the same size as the calling matrix");
	}
	else
	{
		B = new Array<double>(N,N);
	}

	int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N,&N,AT,&N,IPIV,&INFO);
    dgetri_(&N,AT,&N,IPIV,WORK,&LWORK,&INFO);

	__transpose_square<double>(B->data(), AT, N);
	
	luaT_push< Array<double> >(L, B);
	
    delete [] IPIV;
    delete [] WORK;
	delete [] AT;
	
	return 1;
}

// http://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c

int l_matinverse(lua_State* L)
{
	if(luaT_is<dArray>(L, 1)) return l_matinverse_d(L);
// 	if(luaT_is<fArray>(L, 1)) return lT_mat_tri<float,  -1>(L);
// 	if(luaT_is<iArray>(L, 1)) return lT_mat_tri<int,    -1>(L);
// 	if(luaT_is<dcArray>(L, 1)) return lT_mat_tri<doubleComplex, -1>(L);
// 	if(luaT_is<fcArray>(L, 1)) return lT_mat_tri<floatComplex, -1>(L);
	return luaL_error(L, "Unimplemented data type for :matInvese");
}





static int l_matmul_d(lua_State* L)
{
	LUA_PREAMBLE(Array<double>, A, 1);
	LUA_PREAMBLE(Array<double>, B, 2);
	
	if(A->nx != B->ny)
		return luaL_error(L, "Column count of A (%d) does not match row count of B (%d)", A->nx, B->ny);
	
	Array<double>* C = 0;
	if(luaT_is<Array<double> >(L, 3))
		C = luaT_to< Array<double> >(L, 3);
	else
		C = new Array<double>(B->nx, A->ny);
	
	if(C->nx != B->nx || C->ny != A->ny)
		return luaL_error(L, "Size mismatch for destination matrix");	
	
	// trans so we don't have to twiddle stuff for C-f77
	char transa = 'T'; // op(A) = A^T
	char transb = 'T'; // op(B) = B^T
	
	int m = A->ny;
	int n = B->nx;
	int k = A->nx;
	
	double alpha = 1.0;
	
	int LDA = k;
	int LDB = n;
	int LDC = m;
	
	double beta = 0.0; // no C input term

	int res = dgemm_(&transa, &transb, &m, &n, &k, &alpha, A->data(), &LDA, B->data(), &LDB, &beta, C->data(), &LDC);
			   
	__transpose<double>(C->data(), C->data(), C->nx, C->ny);
	
	luaT_push<Array< double> >(L, C);
	return 1;
}


static int l_matmul_f(lua_State* L)
{
	LUA_PREAMBLE(Array<float>, A, 1);
	LUA_PREAMBLE(Array<float>, B, 2);
	
	if(A->nx != B->ny)
		return luaL_error(L, "Column count of A (%d) does not match row count of B (%d)", A->nx, B->ny);
	
	Array<float>* C = 0;
	if(luaT_is<Array<float> >(L, 3))
		C = luaT_to< Array<float> >(L, 3);
	else
		C = new Array<float>(B->nx, A->ny);
	
	if(C->nx != B->nx || C->ny != A->ny)
		return luaL_error(L, "Size mismatch for destination matrix");	
	
	// trans so we don't have to twiddle stuff for C-f77
	char transa = 'T'; // op(A) = A^T
	char transb = 'T'; // op(B) = B^T
	
	int m = A->ny;
	int n = B->nx;
	int k = A->nx;
	
	float alpha = 1.0;
	
	int LDA = k;
	int LDB = n;
	int LDC = m;
	
	float beta = 0.0; // no C input term

	int res = sgemm_(&transa, &transb, &m, &n, &k, &alpha, A->data(), &LDA, B->data(), &LDB, &beta, C->data(), &LDC);
			   
	__transpose<float>(C->data(), C->data(), C->nx, C->ny);
	
	luaT_push<Array< float> >(L, C);
	return 1;
}

int l_matmul(lua_State* L)
{
	if(luaT_is<dArray>(L, 1))
		return l_matmul_d(L);
	if(luaT_is<fArray>(L, 1))
		return l_matmul_f(L);
	return luaL_error(L, "Array.matMul is only implemented for single and double precision arrays");	
}




int l_mat_help(lua_State* L)
{
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_mateigen)
	{
		lua_pushstring(L, "Compute Eigen Values and Eigen Vectors of a Matrix. LAPACK: dgeev");
		lua_pushstring(L, "2 Optional Arrays: The optional arrays will be the target arrays for values and vectors. If none are given then new arrays will be created.");
		lua_pushstring(L, "2 Arrays: The first is an Nx1 array containing values, the second is an NxN array containing vectors. Elements of a single vector share y coordinates (rows).");
		return 3;
	}
	
		
	if(func == l_matinverse)
	{
		lua_pushstring(L, "Compute Inverse of a Matrix. LAPACK: dgetrf + dgetri");
		lua_pushstring(L, "1 Optional Array: The optional arrays will be the target array for the inversion. Cannot be the calling array.");
		lua_pushstring(L, "1 Array: Inverse of the calling array.");
		return 3;
	}
	
	if(func == l_matmul)
	{
		lua_pushstring(L, "Treat arrays like matrices and do Matrix Multiplication on the z=1 layer. BLAS: dgemm/sgemm");
		lua_pushstring(L, "1 Array, 1 Optional Array: The given array will multiply the calling Array, their dimensions must match to allow legal matrix multiplication. If a second Array is supplied the product will be stored in it, otherise a new Array will be created.");
		lua_pushstring(L, "1 Array: The product of the multiplication.");
		return 3;
	}
	
	return 0;
}


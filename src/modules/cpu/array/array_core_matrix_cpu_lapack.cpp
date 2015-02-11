#include "array_core_matrix_cpu_lapack.h"
#include "array.h"
#include <stdlib.h>

#include "lapacke.h"


template <typename T>
Array<T>* getMatrix(lua_State* L, int idx, int nx, int ny, const char* name)
{
    Array<T>* a;
    if(luaT_is< Array<T> >(L, idx))
    {
        a = luaT_to< Array<T> >(L, idx);

        if(a->nx != nx || a->ny != ny)
        {
            luaL_error(L, "Matrix dimension mismatch for `%s', expected (%f x %f) for (%f x %f)", name, ny, nx, a->ny, a->nx);
        }
        return a;
    }

    return new Array<T>(nx,ny);
}



static int l_mateigen_d(lua_State* L)
{
    LUA_PREAMBLE(Array<double>, A, 1);
	
    if(A->nx != A->ny)
        return luaL_error(L, "matrix must be square");
    int n = A->nx;
    
    Array<double>* eval_real = getMatrix<double>(L, 2, 1, n, "EVal(real)");
    Array<double>* evec_real = getMatrix<double>(L, 3, n, n, "EVec(real)");
    Array<double>* eval_imag = getMatrix<double>(L, 4, 1, n, "EVal(imag)");
    Array<double>* evec_imag = getMatrix<double>(L, 5, n, n, "EVec(real)");


    double* wr = eval_real->data();
    double* wi = eval_imag->data();

    double* vr = evec_real->data();
    double* vi = evec_imag->data();


    double* v = new double[n*n];

    lapack_int info = LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'N', 'V', n, 
                                     A->data(), n, wr, wi,
                                     0, n, v, n );

    if( info > 0 ) 
    {
        delete [] v;
        lua_pushnil(L);
        lua_pushstring(L, "The algorithm failed to compute eigenvalues." );
        return 2;
    }

    
    int d = 0;
    for( int i = 0; i < n; i++ ) 
    {
        int j = 0;
        while( j < n ) 
        {
            if( wi[j] == (double)0.0 ) 
            {
                vr[d] = v[i*n+j];
                vi[d] = 0;
                d++;
                j++;
            } 
            else 
            {
                vr[d] = v[i*n+j];
                vi[d] = v[i*n+j+1];
                d++;

                vr[d] = v[i*n+j];
                vi[d] =-v[i*n+j+1];
                d++;
                j += 2;
            }
        }
    }

    delete [] v;

    luaT_push< Array<double> >(L, eval_real);
    luaT_push< Array<double> >(L, evec_real);
    luaT_push< Array<double> >(L, eval_imag);
    luaT_push< Array<double> >(L, evec_imag);
	
    return 4;
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

    lapack_int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    memcpy(B->data(), A->data(), sizeof(double)*N*N);

    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, B->data(), N, IPIV);
	
    LAPACKE_dgetri_work(LAPACK_ROW_MAJOR, N, B->data(),
                        N, IPIV,
                        WORK, LWORK);
	
    luaT_push< Array<double> >(L, B);
	
    delete [] IPIV;
    delete [] WORK;
	
    return 1;
}

// http://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c

int l_matinverse(lua_State* L)
{
    if(luaT_is<dArray>(L, 1)) return l_matinverse_d(L);
    return luaL_error(L, "Unimplemented data type for :matInverse");
}





static int l_matcond_d(lua_State* L)
{
    LUA_PREAMBLE(Array<double>, A, 1);

    char ONE_NORM = '1';
    int NROWS = A->ny;
    int NCOLS = A->nx;
    int LEADING_DIMENSION_A = A->nx;
    int n = LEADING_DIMENSION_A;
    int info;
    double aNorm;
    double rcond;

    double* a = new double[A->nx * A->ny];
    memcpy(a, A->data(), sizeof(double)*A->nx * A->ny);

    lapack_int* ipiv = new lapack_int[n];

    aNorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, ONE_NORM, NROWS, NCOLS, a, 
                           LEADING_DIMENSION_A);
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, NROWS, NCOLS, a, 
                          LEADING_DIMENSION_A, ipiv);
    info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, ONE_NORM, n, a, 
                          LEADING_DIMENSION_A, aNorm, &rcond);

    delete [] ipiv;
    delete [] a;

    if(info != 0)
    {
        lua_pushnil(L);
        switch(info)
        {
        case -4:
            lua_pushstring(L, "Nan in input matrix");
            break;
        case -6:
            lua_pushstring(L, "Nan in norm");
            break;
        default:
            lua_pushstring(L, "Error in condition calculation");
        }
        return 2;
    }

    if(rcond <= 1e-16)
    {
        lua_getglobal(L, "math");
        lua_getfield(L, -1, "huge");
        return 1;
    }

    lua_pushnumber(L, 1.0/rcond);
    return 1;
}

// http://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c

int l_matcond(lua_State* L)
{
    if(luaT_is<dArray>(L, 1)) return l_matcond_d(L);
    return luaL_error(L, "Unimplemented data type for :matCond");
}



static int l_matlinear_system_d(lua_State* L)
{
    LUA_PREAMBLE(Array<double>, A, 1);
    LUA_PREAMBLE(Array<double>, B, 2);

    if(A->nx != A->ny)
        return luaL_error(L, "Calling matrix must be square");

    if(B->ny != A->ny)
        return luaL_error(L, "b matrix dimensions mismatch");

    Array<double>* X = 0;
    if(luaT_is< Array<double> >(L, 3))
    {
        X = luaT_to< Array<double> >(L, 3);
        if(X->nx != B->nx || X->ny != A->ny)
            return luaL_error(L, "X dimensions mismatch");
    }
    else
        X = new Array<double>(B->nx, A->ny);

    int N = A->nx;
    int LDA = N;
    int LDB = B->nx;

    lapack_int* ipiv = new lapack_int[N];

    double* a = A->data();
    double* b = B->data();
    double* x = X->data();

    memcpy(x, b, sizeof(double) * X->nx * X->ny);

    int info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, N, B->nx, a, LDA, ipiv,  x, LDB );

    if( info > 0 ) 
    {
        lua_pushnil(L);
        lua_pushfstring(L, 
                        "The diagonal element of the triangular factor of A, " 
                        "U(%f,%f) is zero, so that A is singular; "
                        "the solution could not be computed.", info, info);

        // delete if needed
        luaT_inc< Array<double> >(X);
        luaT_dec< Array<double> >(X);
        
        return 2;
    }

    luaT_push< Array<double> >(L, X);

    return 1;
}


int l_matlinearsystem(lua_State* L)
{
    if(luaT_is<dArray>(L, 1)) return l_matlinear_system_d(L);
    return luaL_error(L, "Unimplemented data type for :matLinearSystem");
}




template <typename T>
int l_matmul_T(lua_State* L)
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

    for(int x=0; x<C->nx; x++)
        for(int y=0; y<C->ny; y++)
        {
            T s = 0;
            for(int k=0; k<A->nx; k++)
            {
                s += A->get(k,y) * B->get(x,k);
            }
            C->set(x,y,0,s);
        }
	
    luaT_push< Array<T> >(L, C);
    return 1;
}


int l_matmul(lua_State* L)
{
    if(luaT_is<dArray>(L, 1))
        return l_matmul_T<double>(L);
    if(luaT_is<fArray>(L, 1))
        return l_matmul_T<float>(L);
    if(luaT_is<iArray>(L, 1))
        return l_matmul_T<int>(L);
    return luaL_error(L, "Array.matMul is not implemented for complex data types");
}





static int l_matsvd_f(lua_State* L)
{
    LUA_PREAMBLE(Array<float>, A, 1);

    const int M = A->ny;
    const int N = A->nx;
	
    Array<float>* a[3] = {0,0,0};

    int dims[3][2];// = {{M,M}, {M,N}, {N,N}}
    dims[0][0] = M; dims[0][1] = M;
    dims[1][0] = M; dims[1][1] = N;
    dims[2][0] = N; dims[2][1] = N;

    for(int i=0; i<3; i++)
    {
	if(luaT_is<Array<float> >(L, 2+i))
	    a[i] = luaT_to< Array<float> >(L, 2+i);
	else
	    a[i] = new Array<float>(dims[i][1], dims[i][0]);
	
	if(a[i]->nx != dims[i][1] || a[i]->ny != dims[i][0])
	    return luaL_error(L, "Size mismatch for destination matrix");	
    }

    Array<float>* U = a[0];
    Array<float>* SIGMA = a[1];
    Array<float>* VT = a[2];

// lapack_int LAPACKE_sgesdd (int matrix_order, char jobz, lapack_int m, lapack_int n, float *a, lapack_int lda, float *s, float *u, lapack_int ldu, float *vt, lapack_int ldvt)
    int info;
    info = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', M, N, A->data(), N, SIGMA->data(), U->data(), M, VT->data(), N);
    if (info) // handle error conditions here
    {
	//return luaL_error(L, "The algorithm computing SVD failed to converge.");
	lua_pushnil(L);
	lua_pushstring(L, "The algorithm computing SVD failed to converge.");
	return 2;
    }



    // SIGMA is a row of data, need to make it a diagonal matrix
    const int SX = SIGMA->nx;
    for(int x=1; x<SX && x<SIGMA->ny; x++)
    {
	SIGMA->data()[x * SX + x] = SIGMA->data()[x];
	SIGMA->data()[x] = 0;
    }

    luaT_push<fArray>(L, a[0]);
    luaT_push<fArray>(L, a[1]);
    luaT_push<fArray>(L, a[2]);

    return 3;
}


static int l_matsvd_d(lua_State* L)
{
    LUA_PREAMBLE(Array<double>, A, 1);

    const int M = A->ny;
    const int N = A->nx;
	
    Array<double>* a[3] = {0,0,0};

    int dims[3][2];// = {{M,M}, {M,N}, {N,N}}
    dims[0][0] = M; dims[0][1] = M;
    dims[1][0] = M; dims[1][1] = N;
    dims[2][0] = N; dims[2][1] = N;

    for(int i=0; i<3; i++)
    {
	if(luaT_is<Array<double> >(L, 2+i))
	    a[i] = luaT_to< Array<double> >(L, 2+i);
	else
	    a[i] = new Array<double>(dims[i][1], dims[i][0]);
	
	if(a[i]->nx != dims[i][1] || a[i]->ny != dims[i][0])
	    return luaL_error(L, "Size mismatch for destination matrix");	
    }

    Array<double>* U = a[0];
    Array<double>* SIGMA = a[1];
    Array<double>* VT = a[2];

    int info;
    info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', M, N, A->data(), N, SIGMA->data(), U->data(), M, VT->data(), N);
    if (info) // handle error conditions here
    {
	//return luaL_error(L, "The algorithm computing SVD failed to converge.");	
	lua_pushnil(L);
	lua_pushstring(L, "The algorithm computing SVD failed to converge.");
	return 2;
    }

    // SIGMA is a row of data, need to make it a diagonal matrix
    const int SX = SIGMA->nx;
    for(int x=1; x<SX && x<SIGMA->ny; x++)
    {
	SIGMA->data()[x * SX + x] = SIGMA->data()[x];
	SIGMA->data()[x] = 0;
    }

    luaT_push<dArray>(L, a[0]);
    luaT_push<dArray>(L, a[1]);
    luaT_push<dArray>(L, a[2]);

    return 3;
}





int l_matsvd(lua_State* L)
{
    if(luaT_is<dArray>(L, 1))
	return l_matsvd_d(L);
    if(luaT_is<fArray>(L, 1))
	return l_matsvd_f(L);
    return luaL_error(L, "Array.matSVD is only implemented for single and double precision arrays");
}




int l_mat_lapack_help(lua_State* L)
{
    lua_CFunction func = lua_tocfunction(L, 1);

    if(func == l_matsvd)
    {
        lua_pushstring(L, "Compute Singular Value Decomposition of a Matrix. A = U * SIGMA * transpose(V). A is MxN, U is MxM, SIGMA is MxN and V is NxN. LAPACK: dgesdd");
        lua_pushstring(L, "3 Optional Arrays: The optional arrays will be the target arrays decomposition, dimensions must match. If no arrays are given then new arrays will be created. The first is the U matix, the second is the SIGMA matix and the third will be the transpose of V (not V itself).");
        lua_pushstring(L, "3 Arrays: The first is the U matix, the second is the SIGMA matix and the third will be the transpose of V (not V itself).");
        return 3;

    }
	
    if(func == l_mateigen)
    {
        lua_pushstring(L, "Compute Eigen Values and Eigen Vectors of a Matrix. LAPACK: dgeev");
        lua_pushstring(L, "4 Optional Arrays: The optional arrays will be the target arrays for values and vectors. If no arrays are given then new arrays will be created. The first is the real part of the eigen values, the second is the NxN array which will contain the eigen vectors and the third is the imaginary part of the values and the forth is the imaginary part of the eigen vectors.");
        lua_pushstring(L, "4 Arrays: The first is an Nx1 array containing the real values, the second is an NxN array containing vectors. The third contains the imaginary parts of the values and the forth is the imaginary parts of the vectors. Elements of a single vector share y coordinates (rows). The order of the return values are awkward. If something changes in the future the method name will change so errors are not silent.");
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
	
    if(func == l_matcond)
    {
        lua_pushstring(L, "Compute the 1-norm condition number of a matrix.");
        lua_pushstring(L, "1 Array: Matrix");
        lua_pushstring(L, "1 Number: 1-norm condition number");
        return 3;
    }
	
    if(func == l_matlinearsystem)
    {
        lua_pushstring(L, "Solve linear system A X = B where the calling array is A. Example:\n<pre>x = A:matLinearSystem(b)</pre>");
        lua_pushstring(L, "1 Array, 1 optional Array: B vector (or matrix), X vector (or matrix).");
        lua_pushstring(L, "1 Array: Solution X.");
        return 3;
    }
	

    return 0;
}


#include<iostream>
#include<time.h>
#include<sys/time.h>
#include<iomanip>
//#include<x86intrin.h>
#include "omp.h"

using namespace std;
#define REAL_T double 

void printFlops(int A_height, int B_width, int B_height, timeval start, timeval end  ){
	float time_use = 0;
    time_use = (end.tv_sec - start.tv_sec)  + (end.tv_usec - start.tv_usec)/(float)1000000;//微秒
    cout<<"SECOND:\t"<<time_use<<"s\n";
}

// void printFlops(int A_height, int B_width, int B_height, clock_t start,clock_t end ){
// 	cout<<"SECOND:\t"<<(end - start)/CLOCKS_PER_SEC<<"."<<(end - start)%CLOCKS_PER_SEC<<"\t\t";
// 	REAL_T flops = ( 2.0 * A_height * B_width * B_height ) / 1E9 /((end -  start)/(CLOCKS_PER_SEC * 1.0));
// 	cout<<"GFLOPS:\t"<<flops<<endl;
// }

void initMatrix( int n, REAL_T *A, REAL_T *B, REAL_T *C ){
	for( int i = 0; i < n; ++i )
		for( int j = 0; j < n; ++j ){
			A[i+j*n] = (i+j + (i*j)%100 ) %100;
			B[i+j*n] = ((i-j)*(i-j) + (i*j)%200 ) %100;
			C[i+j*n] = 0;
		}
}

void dgemm( int n, REAL_T *A, REAL_T *B, REAL_T *C){
	for( int i = 0; i < n; ++i )
		for( int j = 0; j < n; ++j ){
			REAL_T cij = C[i+j*n];
			for( int k = 0; k < n; k++ ){
				cij += A[i+k*n] * B[k+j*n];
			}
			C[i+j*n] = cij;
		}
}

// void avx_dgemm(int n, REAL_T *A, REAL_T *B, REAL_T *C){
// 	for( int i = 0; i < n; i+=4 )
// 		for( int j = 0; j < n; ++j ){
// 			__m256d cij = _mm256_load_pd( C+i+j*n );
// 			for( int k = 0; k < n; k++ ){
// 				//cij += A[i+k*n] * B[k+j*n];
// 				cij = _mm256_add_pd( 
// 						cij, 
// 						_mm256_mul_pd( _mm256_load_pd(A+i+k*n),  _mm256_load_pd(B+i+k*n) )
// 						);
// 			}
// 			_mm256_store_pd(C+i+j*n,cij);
// 		}
// }

#define UNROLL (4)

// void pavx_dgemm(int n, REAL_T *A, REAL_T *B, REAL_T *C){
// 	for( int i = 0; i < n; i+=1*UNROLL )
// 		for( int j = 0; j < n; ++j ){
// 			REAL_T cij = C[i+j*n];
// 			for( int k = 0; k < n; k++ ){
// 				//cij += A[i+k*n] * B[k+j*n];
// 				for( int x = 0; x <UNROLL; ++x)
// 					cij=cij+A[i+(x+k)*n]*B[x+k+j*n];
// 			}
// 				C[i+j*n]=cij;
// 		}
// }
void pavx_dgemm(int n, REAL_T *A, REAL_T *B, REAL_T *C){
	for( int i = 0; i < n; i+=UNROLL)
		for( int j = 0; j < n-4; ++j ){
			REAL_T cij[4];
			for( int x = 0; x < UNROLL; ++x)
				cij[x]=C[i+(j+x)*n];
			for( int k = 0; k < n-4; k++ ){
				//cij += A[i+k*n] * B[k+j*n];
				/*cij = _mm256_add_pd( 
						cij, 
						_mm256_mul_pd( _mm256_load_pd(A+i+k*n),  _mm256_load_pd(B+i+k*n) )
						);*/
				for( int x = 0; x <UNROLL; ++x)
					cij[x]=cij[x]+A[i+(x+k)*n]*B[k+j*n];
			}
			for( int x = 0; x < UNROLL; ++x)
				C[i+(x+j)*n]=cij[x];
		}
}
#define BLOCKSIZE (32)
void do_block( int n, int si, int sj, int sk, REAL_T *A, REAL_T *B, REAL_T *C){
	for( int i = si; i < si + BLOCKSIZE; i+=UNROLL )
		for( int j = sj; j < sj + BLOCKSIZE; ++j){
			REAL_T c[4];
			for( int x = 0; x < UNROLL; ++x )
				c[x] =C[i+(j+x)*n];

			for( int k = sk; k < sk + BLOCKSIZE; ++k ){
				REAL_T b =B[k+j*n];
				for( int x = 0; x <UNROLL; ++x)
					c[x] =c[x]+A[i+(x+k)*n]*b;
			}

			for( int x = 0; x < UNROLL; ++x)
				C[i+(x+j)*n]=c[x];
		}
}
void block_gemm(int n, REAL_T *A, REAL_T *B, REAL_T *C){
	for( int sj = 0; sj <n; sj+=BLOCKSIZE)
		for( int si = 0; si <n; si+=BLOCKSIZE)
			for( int sk = 0; sk <n; sk+=BLOCKSIZE)
				do_block( n, si, sj, sk, A, B, C);
}
void omp_gemm(int n, REAL_T *A, REAL_T *B, REAL_T *C){
#pragma omp parallel for
	for( int sj = 0; sj <n; sj+=BLOCKSIZE)
	{
		for( int si = 0; si <n; si+=BLOCKSIZE)
			for( int sk = 0; sk <n; sk+=BLOCKSIZE)
				do_block( n, si, sj, sk, A, B, C);
	}

}

int main()
{
	REAL_T *A, *B, *C;
	struct timeval start;
    struct timeval end;
	int n = 1024;
	A = new REAL_T[n*n];
	B = new REAL_T[n*n];
	C = new REAL_T[n*n];
	//initMatrix(n, A, B, C);
	
	cout<< "origin caculation begin...\n";
    gettimeofday(&start, NULL);
	dgemm( n, A, B, C );
	gettimeofday(&end, NULL);
	//cout <<(stop - start)/CLOCKS_PER_SEC<<"."<<(stop - start)%CLOCKS_PER_SEC<<"\t\t";
	printFlops(n, n, n, start, end);

	initMatrix(n, A, B, C);
	cout<< "parallel AVX caculation begin...\n";
	gettimeofday(&start, NULL);
	pavx_dgemm( n, A, B, C );
	gettimeofday(&end, NULL);
	//cout <<(stop - start)/CLOCKS_PER_SEC<<"."<<(stop - start)%CLOCKS_PER_SEC<<"\t\t";
	printFlops(n, n, n, start, end);

	initMatrix(n, A, B, C);
	cout<< "blocked AVX caculation begin...\n";
	//start = clock();
	gettimeofday(&start, NULL);
	block_gemm( n, A, B, C );
	gettimeofday(&end, NULL);
	//stop = clock();
	//cout <<(stop - start)/CLOCKS_PER_SEC<<"."<<(stop - start)%CLOCKS_PER_SEC<<"\t\t";
	printFlops(n, n, n, start, end);

	// initMatrix(n, A, B, C);
	// cout<< "OpenMP blocked AVX caculation begin...\n";
	// starts = clock();
	// //gettimeofday(&start, NULL);
	// omp_gemm( n, A, B, C );
	// //gettimeofday(&end, NULL);
	// stops = clock();
	// cout <<(stops - starts)/CLOCKS_PER_SEC<<"."<<(stops - starts)%CLOCKS_PER_SEC<<"\t\t";
	// //printFlops(n, n, n, start, end);

	initMatrix(n, A, B, C);
	cout<< "OpenMP blocked AVX caculation begin...\n";
	//start = clock();
	gettimeofday(&start, NULL);
	omp_gemm( n, A, B, C );
	gettimeofday(&end, NULL);
	//stop = clock();
	//cout <<(stop - start)/CLOCKS_PER_SEC<<"."<<(stop - start)%CLOCKS_PER_SEC<<"\t\t";
	printFlops(n, n, n, start, end);
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#define SPMAX(a,b) (((a) > (b)) ? (a) : (b))
#define SPLONG long

typedef struct CSparse
{
    int nzmax;
    int m;
    int n;
    int *p;
    int *i;
    float *x;
    int nz;
} sp_si;

#define sp sp_si

void *sp_calloc (int n, size_t size) {
    return (calloc ( SPMAX(n, 1), size));
}

/* wrapper for malloc */
void *sp_malloc (SPLONG n, size_t size)
{
    return (malloc (SPMAX (n,1) * size)) ;
}

/* wrapper for free */
void *sp_free (void *p)
{
    if (p) free (p) ;       /* free p if it is not already NULL */
    return (NULL) ;         /* return NULL to simplify the use of cs_free */
}

/* free a sparse matrix */
sp *sp_spfree (sp *A)
{
    if (!A) return (NULL) ;     /* do nothing if A already NULL */
    //sp_free (A->p) ;
    //sp_free (A->i) ;
    //sp_free (A->x) ;
    return ((sp *) sp_free (A)) ;   /* free the cs struct and return NULL */
}

sp *sp_spalloc (SPLONG m, SPLONG n, SPLONG nzmax, SPLONG values, SPLONG triplet)
{
    sp *A = sp_calloc (1, sizeof (sp)) ;    /* allocate the sp struct */
    if (!A) return (NULL) ;                 /* out of memory */
    A->m = m ;                              /* define dimensions and nzmax */
    A->n = n ;
    A->nzmax = nzmax = SPMAX (nzmax, 1) ;
    A->nz = triplet ? 0 : -1 ;              /* allocate triplet or comp.col */
    A->i = sp_malloc (triplet ? nzmax : m+1, sizeof (SPLONG)) ;
    A->p = sp_malloc (nzmax, sizeof (SPLONG)) ;
    A->x = values ? sp_malloc (nzmax, sizeof (SPLONG)) : NULL ;
    return ((!A->p || !A->i || (values && !A->x)) ? sp_spfree (A) : A) ;
}

sp * create_csr(int m, int n, int nz, int *i, int *p, float *x) {
    sp *A;
    A = sp_spalloc(m, n, nz, 1, 0);
    A -> i = i;
    A -> p = p;
    A -> x = x;
    return A;
}

int gaxpy(sp *A, const float *x, float *y) {
    int m, *Ap, *Ai; 
    float *Ax, temp;
    m = A -> m; Ap = A -> p; Ai = A -> i; Ax = A -> x;
    for ( int i = 0; i < m; i ++ ) {
        temp = y[i];
        for (int j = Ai[i]; j < Ai[i + 1]; j++ ) {
            temp += Ax[j] * x[ Ap[j] ];
        }
        y[i] = temp;
    }
    return 0;
}

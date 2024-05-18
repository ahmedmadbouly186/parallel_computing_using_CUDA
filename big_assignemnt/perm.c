#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
int* kthPermutation(const int n, int i)
{
   int j, k = 0;
   int *fact = (int *)calloc(n, sizeof(int));
   int *perm = (int *)calloc(n, sizeof(int));

   // compute factorial numbers
   fact[k] = 1;
   while (++k < n)
      fact[k] = fact[k - 1] * k;

   // compute factorial code
   for (k = 0; k < n; ++k)
   {
      perm[k] = i / fact[n - 1 - k];
      i = i % fact[n - 1 - k];
   }

   // readjust values to obtain the permutation
   // start from the end and check if preceding values are lower
   for (k = n - 1; k > 0; --k)
      for (j = k - 1; j >= 0; --j)
         if (perm[j] <= perm[k])
            perm[k]++;

   // print permutation
//    for (k = 0; k < n; ++k)
//       printf("%d ", perm[k]);
//    printf("\n");

   free(fact);
   return perm;
}

void main(int argc,char * arrgv){
    int n=4;
    int k=5;
    int total_perm=1;
    for(int i=1;i<=n;i++){
        total_perm*=i;
    }
    for(int i=0;i<total_perm;i++){
        int *perm=kthPermutation(n,i);
       
    }
}
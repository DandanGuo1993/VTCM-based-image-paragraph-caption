#include<stdio.h>
#include<stdlib.h>
#include<math.h>
// Chaojie 2018_1_15

int Binary_Search(double *probvec, double prob, int K)
{
	int kstart, kend, kmid;
	// K : the length of probvec
	if (prob <= probvec[0])
		return(0);
	else
	{
		for(kstart = 1, kend = K-1;;)
		{
			if (kstart >= kend)
				return(kend);
			else
			{
				kmid = (kstart + kend)/2;
				if (probvec[kmid-1]>=prob)
					kend = kmid - 1;
				else if (probvec[kmid]<prob)
					kstart = kmid + 1;
				else
					return(kmid);
			}
		}
	}
	return(kmid);
}
	
void Multi_Sample(double* X, double* Phi, double* Theta, double* D, double* XKJ, int V, int K, int J, int V1, int K1, int K3)  // V1 K1 denotes image scale , feature map scale
{
	double* probvec = (double*)malloc(K * sizeof(double));
	
//	if (probvec == NULL)
//		printf("Malloc Error, No space!\n");

	for (int v=0;v<V;v++)                                 // 0 - v-1
	{
		for (int j=0;j<J;j++)			      // 0 - j-1
		{
			if(X[v*J+j]<0.5)
				continue;
			else
			{
				double cumsum = 0.0;
				for(int k=0;k<K;k++)
				{
					cumsum += Phi[v*K + k] * Theta[k*J + j];
					probvec[k] = cumsum;
				}

				for (int token = 0; token<X[v*J + j]; token++)
				{
					double probrnd = ((double)(rand()) / RAND_MAX) * cumsum;
					int Embedding_K = Binary_Search(probvec,probrnd,K);
				//	printf("%d %d\n",v,j);
					//XVK[v*K + Embedding_K] += 1;
				//	int re_x = (v+1)%V1;
				//	int re_y = (Embedding_K+1)%K1;
				//	int dx = (int)( re_x ) - (int)( re_y );
				//	int dy = (int)( (v+1 - re_x) /V1 ) - (int)( (Embedding_K+1 - re_y) /K1 );
				//	printf("%d %d\n",dx,dy);
				//	if (dx<0 || dy<0)
				//		continue;

					int dx = (int)( (v)%V1 ) - (int)( (Embedding_K)%K1 );
					int dy = (int)( ceil((v+1.0) /(double)(V1) )) - (int)( ceil((Embedding_K+1.0) /(double)(K1)) );
				//	printf("%d %d %d %d\n",v,Embedding_K,dx,dy);
					D[dx*K3 + dy] += 1;
					XKJ[Embedding_K*J + j] += 1;
				}
			}		
		}
	}
	free(probvec);
}




#include <iostream>
#include <stdio.h>

//https://blog.csdn.net/mhl_1208980380/article/details/54230369

using namespace std;

int test1(int N)
{
	int d,e=2,i=0,j,k=10,m;
	N++;
	int *a=new int[2*N],*b=new int[N],*c=new int[2*N];
	while(++i<2*N)a[i]=b[i/2]=c[i]=0;
	for(cout<<"0.6",a[b[1]=m=2]=4;--N>1;)
		for(b[m]=e,d=0,j=k;--j+1&&!d;)
		{
			for(c[i=2*m]=j*j%k,e=j*j/k;--i>m-2;e=d/k)c[i]=(d=e+b[i-m+1]*j)%k;
			for(i=d=m-1;i<2*m&&a[i]<=c[i];i++)(a[i]<c[i])?d=0:1;
			if(d)for(e=j<<1,cout<<j,i=1+2*m++;--i>m-2;)if((a[i]-=c[i])<0)a[i]+=k,a[i-1]--;
		}
	delete []a,delete []b,delete []c,cin.ignore(),cin.ignore();
	return 0;
}

int test2(int N) {
    long i = 0,j,k,p,d = 10000,f = 1180,m=1;
    N /= 4;
    cout <<"\n0.6180";
    long *a = new long [2*N+1];
    long *b = new long [2*N+1];
    long *c = new long [2*N+1];
    while (++i<2*N) {
        a[i] = b[i] = c[i] = 0;
    }
    for (a[b[1]=2]=7600;--N;)
        for(b[++m]=f*2, printf("%04ld", b[m+1] = f = int(((a[m-1]*d+a[m])*d+a[m +1])/22360.679775)), j=p=0, i=2*m+1; --i>m-1;j=k/d)
            p=(a[i]-=(c[i]=(k=b[i-m+1]*f+j)%d)+p) < 0 ? a[i]+=d, 1:0;
    delete []a;
    delete []b;
    delete []c;
    cout << "\nThe End";
    cin.ignore();
    cin.ignore();
    return 0;
}

int test3(int N)
{
	int d=10000,e,f=1180,g,i=0,j,m = 1;
	N=N/4+1;
  cout<<"1.6180";
	long *a = new long [N];
  long *b = new long [2*N];
  long *c = new long [2*N];
	while(++i<2*N) {
      a[i/2]=b[i]=c[i]=0;
  }
	for (b[a[m]=2]=7600; --N>1; printf("%04d",f)) {
		for (a[++m]=f*2,j=e=d,f=0;(e-f)&&j;) {
			for (g=(e+f)/2,i=2*m;--i>m-2;) {
          c[i]=a[i-m+1]*g;
      }
			for (c[i=2*m]=g*g;i>m-1;c[i--]%=d) {
          c[i-1]+=c[i]/d;
      }
			for (*a=i=m-1,j=g-f;i<2*m&&*a;i++) {
          b[i]>c[i] ? f=g,*a=0 : b[i]<c[i] ? e=g,*a=0:1;
      }
		}
		for (i=2*m;i>m-2;i--)if((b[i]-=c[i])<0) {
        b[i]+=d,b[i-1]--;
    }
	}
  cout << "\n";
	delete []a,delete []b,delete []c,cin.ignore(),cin.ignore();
	return 0;
}

int main(int argc, char *argv[]) {
    // 生成的数字少一个 9
    int N = 2105;
    test3(N);
}

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
float train[][2]={
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8}
};
#define train_count (sizeof(train)/sizeof(train[0]))
float rand_float()
{
    return ((float)rand())/((float)RAND_MAX);
}

float cost_fucntion(float w)
{
float result=0.0f;
    for(size_t i=0;i<train_count;i++)
    {
        float x=train[i][0];
        float y=x*w;
        float d=y-train[i][1];
        result+=d*d;
    }
    result /=train_count;
    return result;
}
float dcost(float w)
{
    float result=0.f;
    size_t n=train_count;
    for(size_t i=0;i<n;i++)
    {
        float x=train[i][0];
        float y=train[i][1];
        result+=2*(x*w-y)*x;
    }
    result/=n;
    return result;
}

int main()
{
    srand(time(0));
    float w=rand_float()*10.0f;
    float eps=1e-3;
    float rate=1e-1;
    printf("%f %f\n",cost_fucntion(w),w);
    for(size_t i=0;i<30;i++)
    {
        // float c=cost_fucntion(w);
      float dw=dcost(w);
   w-=rate*dw;
   printf("%f w=%f\n",cost_fucntion(w),w);
  } 
  printf("-----------\n");
  printf("w=%f\n",w); 
}
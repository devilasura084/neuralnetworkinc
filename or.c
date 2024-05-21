#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
typedef float sample[3];
float sigmoidf(float x)
{
 return 1.0f/(1.0f + expf(-x));
}
sample or_train[]={
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1},
};
sample and_train[]={
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1},
};
sample nand_train[]={
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};
sample *train=or_train;
size_t train_count= 4;

float rand_float()
{
    return ((float)rand())/((float)RAND_MAX);
}
float cost_fucntion(float w1,float w2,float b)
{
float result=0.0f;
    for(size_t i=0;i<train_count;i++)
    {
        float x1=train[i][0];
        float x2=train[i][1];
        float y=sigmoidf(x1*w1+x2*w2+b);
        float d=y-train[i][2];
        result+=d*d;
    }
    result /=train_count;
    return result;
}
void gcost(float w1,float w2,float b,float *dw1,float *dw2,float *db)
{
    *dw1=0;
    *dw2=0;
    *db=0;
    for(size_t i=0;i<train_count;i++)
    {
        float xi=train[i][0];
        float yi=train[i][1];
        float zi=train[i][2];
        float ai=sigmoidf(xi*w1 +yi*w2 +b);
        float di=2*(ai-zi)*ai*(1-ai);
        *dw1+=di*xi;
        *dw2+=di*yi;
        *db+=di;
    }
    *dw1/=train_count;
    *dw2/=train_count;
    *db/=train_count;
}
int main()
{

    srand(time(0));
float w1=rand_float();
float w2=rand_float();
float b=rand_float();
printf("w1 = %f w2 = %f b=%f\n",w1,w2,b);
float eps=1e-3;
float rate=1e-1;
for(int i=0;i<10000;i++)
{
    float dw1,dw2,db;
    float c=cost_fucntion(w1,w2,b);
    printf("c=%f w1 = %f w2 = %f b= %f \n",c,w1,w2,b);
    gcost(w1,w2,b,&dw1,&dw2,&db);
    w1-=rate * dw1;
    w2-=rate * dw2;
    b-=rate*db;
}
printf("c=%f w1 = %f w2 = %f b=%f \n",cost_fucntion(w1,w2,b),w1,w2,b);
for(int i=0;i<2;i++)
{
    for(int j=0;j<2;j++)
    {
        printf("%zu | %zu = %f\n",i,j,sigmoidf(i*w1+j*w2+b));
    }
}
}
#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>
#define eps 1e-1
#define rate 1
float td[]={
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};
int main(void)
{
    srand(69);
    size_t stride =3;
    size_t n=sizeof(td)/sizeof(td[0])/stride;
    Mat ti={
        .rows =n,
        .cols =2,
        .stride =stride,
        .es =td
    };
    Mat to={
        .rows =n,
        .cols=1,
        .stride=stride,
        .es =td+2,
    };
    
    size_t arch[]={2,2,1};
    NN nn;
    NN gg;
    nn=nn_alloc(ARRAy_LEN(arch),arch);
    gg=nn_alloc(ARRAy_LEN(arch),arch);
    nn_rand(nn,0,1);
    for(size_t i=0;i<5;i++)
    {
#if 1
    nn_finit_diff(nn,gg,ti,to,eps);
    nn_learn(nn,gg,rate);
#else
    
    nn_backprop(nn,gg,to,ti);
    nn_learn(nn,gg,rate);
    printf("%f\n",nn_cost(nn,ti,to));
#endif
           
    }
    NN_PRINT(nn);
    for(size_t i=0;i<2;i++)
    {
        for(size_t j=0;j<2;j++)
        {
            MAT_AT(NN_INPUT(nn),0,0)=i;
            MAT_AT(NN_INPUT(nn),0,1)=j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n",i,j,MAT_AT(NN_OUTPUT(nn),0,0));
        }
    }
    return 0;

}
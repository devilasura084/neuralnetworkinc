#include <stddef.h>
#include <stdio.h>
#include <math.h>
#ifndef NN_H_
#define NN_H_
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif
#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif
typedef struct{
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
}Mat;
#define MAT_PRINT(m) mat_print(m,#m,0)
#define MAT_AT(m,i,j) (m).es[(i)*(m).stride + (j)]
float sigmoidf(float x)
{
    return 1.f/(1.f+expf(-x));
}
void mat_sig(Mat m)
{
    for(size_t i=0;i<m.rows;i++)
    {
        for(size_t j=0;j<m.cols;j++)
        {
           MAT_AT(m,i,j)=sigmoidf(MAT_AT(m,i,j));
        }
    }
}
Mat mat_row(Mat m,size_t row)
{
    return (Mat){
        .rows=1,
        .cols=m.cols,
        .stride=m.stride,
        .es=&MAT_AT(m,row,0)};
}
void mat_copy(Mat dst,Mat src)
{
    NN_ASSERT(dst.rows==src.rows);
    NN_ASSERT(dst.cols==src.cols);
    for(size_t i=0;i<dst.rows;i++)
    {
        for(size_t j=0;j<dst.cols;j++)
        {
            MAT_AT(dst,i,j)=MAT_AT(src,i,j);
        }
    }
}
Mat mat_alloc(size_t rows,size_t cols)
{
    Mat m;
    m.rows=rows;
    m.cols=cols;
    m.stride=cols;
    m.es=NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es !=NULL);
    return m;
}
void mat_dot(Mat dst,Mat a,Mat b)
{
    NN_ASSERT(a.cols==b.rows);
    size_t n=a.cols;
    NN_ASSERT(dst.rows==a.rows);
    NN_ASSERT(dst.cols==b.cols);
    for(size_t i=0;i<dst.rows;++i)
    {
        for(size_t j=0;j<dst.cols;++j)
        {
            MAT_AT(dst,i,j)=0;
            for(size_t k=0;k<n;++k)
            {
                MAT_AT(dst,i,j)+=MAT_AT(a,i,k)*MAT_AT(b,k,j);
            }
        }
    }
}
void mat_sum(Mat dst,Mat a)
{
    NN_ASSERT(dst.rows==a.rows);
    NN_ASSERT(dst.cols==a.cols);
    for(size_t i=0;i<dst.rows;i++)
    {
        for(size_t j=0;j<dst.cols;j++)
        {
            MAT_AT(dst,i,j)+=MAT_AT(a,i,j);
        }
    }
}
float rand_float(void)
{
    return ((float)rand())/((float)RAND_MAX);
}
void mat_rand(Mat m,float low,float high)
{
     for(size_t i=0;i<m.rows;i++)
    {
        for(size_t j=0;j<m.cols;j++)
        {
            MAT_AT(m,i,j)=rand_float()*(high-low)+low;
        }
    }
}
void mat_fill(Mat m,float x)
{
    for(size_t i=0;i<m.rows;i++)
    {
        for(size_t j=0;j<m.cols;j++)
        {
            MAT_AT(m,i,j)=x;
        }
    }
}
void mat_print(Mat m,char *name,size_t padding)
{
    printf("%*s%s = [\n",(int)padding,"",name);
    for(size_t i=0;i<m.rows;i++)
    {
        printf("%*s",(int)padding,"");
        for(size_t j=0;j<m.cols;j++)
        {
            printf("  %f",MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("%*s ] \n",(int)padding,"");
}
#endif 

#ifdef NN_IMPLEMENTATION
#define ARRAy_LEN(xs) sizeof(xs)/sizeof(xs[0])
typedef struct {
    size_t count;
    Mat *w;
    Mat *b;
    Mat *a;//activation = count+1
}NN;
NN nn_alloc(size_t arch_count,size_t *arch)
{
    NN_ASSERT(arch_count>0);
   NN m;
    m.count=arch_count-1;
    m.w=NN_MALLOC(sizeof(*m.w)*m.count);
    NN_ASSERT (m.w!=NULL);
    m.b=NN_MALLOC(sizeof(*m.b)*m.count);
    NN_ASSERT (m.b!=NULL);
     m.a=NN_MALLOC(sizeof(*m.a)*(m.count+1));
    NN_ASSERT (m.a!=NULL);

    m.a[0]=mat_alloc(1,arch[0]);
    for(int i=1;i<arch_count;i++)
    {
        m.w[i-1]=mat_alloc(m.a[i-1].cols,arch[i]);
        m.b[i-1]=mat_alloc(1,arch[i]);
        m.a[i]=mat_alloc(1,arch[i]);
    }
    return m;

}
#define NN_PRINT(nn) nn_print(nn,#nn)
#define NN_INPUT(nn) (nn).a[0]
#define NN_OUTPUT(nn) (nn).a[(nn).count]
void nn_print(NN m,const char *name)
{
    char buf[256];
    printf("%s = [\n");
    Mat *w=m.w;
    Mat *b=m.b;
    for(size_t i=0;i<m.count;i++)
    {
        snprintf(buf,sizeof(buf),"w[%zu]",i);
        mat_print(w[i],buf,4);
        snprintf(buf,sizeof(buf),"b[%zu]",i);
        mat_print(b[i],buf,4);
    }

    printf("]\n");
}
void nn_rand(NN m,float low,float high)
{
    for(size_t i=0;i<m.count;i++)
    {
        mat_rand(m.w[i],low,high);
        mat_rand(m.b[i],low,high);
    }
}
void nn_forward(NN nn)
{
    for(size_t i=0;i<nn.count;i++)
    {
        mat_dot(nn.a[i+1],nn.a[i],nn.w[i]);
        mat_sum(nn.a[i+1],nn.b[i]);
        mat_sig(nn.a[i+1]);
    }
}
float nn_cost(NN nn,Mat ti,Mat to)
{
     assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t p=ti.rows;
    size_t q=to.cols;
    float c=0.f;
    for(size_t i=0;i<p;i++)
    {
        Mat x=mat_row(ti,i);
        Mat y=mat_row(to,i);
        mat_copy(NN_INPUT(nn),x);
        nn_forward(nn);

        size_t q=to.cols;
        for(size_t j=0;j<q;j++)
        {
            float d=MAT_AT(NN_OUTPUT(nn),0,j)-MAT_AT(y,0,j);
            c+=d*d;
        }
    }
    return c/p;
}
void nn_finit_diff(NN m,NN g,Mat ti,Mat to,float eps)
{
    float saved;
    float c=nn_cost(m,ti,to);
    for(size_t k=0;k<m.count;k++)
    {
      for(size_t i=0;i<m.w[k].rows;i++)
      {
        for(size_t j=0;j<m.w[k].cols;j++)
        {
            saved=MAT_AT(m.w[k],i,j);
            MAT_AT(m.w[k],i,j)+=eps;
            MAT_AT(g.w[k],i,j)=(nn_cost(m,ti,to)-c)/eps;
            MAT_AT(m.w[k],i,j)=saved;
        }
      }
      for(size_t i=0;i<m.b[k].rows;i++)
      {
        for(size_t j=0;j<m.b[k].cols;j++)
        {
            saved=MAT_AT(m.b[k],i,j);
            MAT_AT(m.b[k],i,j)+=eps;
            MAT_AT(g.b[k],i,j)=(nn_cost(m,ti,to)-c)/eps;
            MAT_AT(m.b[k],i,j)=saved;
        }
      }
    }
}
void nn_learn(NN m,NN g,float rate)
{
    for(size_t k=0;k<m.count;k++)
    {
      for(size_t i=0;i<m.w[k].rows;i++)
      {
        for(size_t j=0;j<m.w[k].cols;j++)
        {
            MAT_AT(m.w[k],i,j)-=rate*MAT_AT(g.w[k],i,j);
        }
      }
      for(size_t i=0;i<m.b[k].rows;i++)
      {
        for(size_t j=0;j<m.b[k].cols;j++)
        {
            MAT_AT(m.b[k],i,j)-=rate*MAT_AT(g.b[k],i,j);
        }
      }
    }
}
void nn_zero(NN nn)
{
    for(size_t i=0;i<nn.count;i++)
    {
        mat_fill(nn.w[i],0);
        mat_fill(nn.b[i],0);
        mat_fill(nn.a[i],0);
    }
    mat_fill(nn.a[nn.count],0);
}
void nn_backprop(NN n,NN g,Mat to,Mat ti){
    NN_ASSERT(ti.rows==to.rows);
    size_t nn=ti.rows;
    NN_ASSERT(NN_OUTPUT(n).cols==to.cols);
    nn_zero(g);
    //i-current sample
    //l-current layer
    //j-current activation
    //k-previous activation
   for(size_t i=0;i<nn;i++)
   {
    mat_copy(NN_INPUT(n),mat_row(ti,i));
    nn_forward(n);
    for(size_t j=0;j<=n.count;j++)
    {
        mat_fill(g.a[j],0);
    }
    for(size_t j=0;j<to.cols;j++)
    {
       MAT_AT(NN_OUTPUT(g),0,j)= MAT_AT(NN_OUTPUT(n),0,j)-MAT_AT(to,i,j);
    }
    for(size_t l=n.count;l>0;--l)
    {
        for(size_t j=0;j<n.a[l].cols;j++)
        {
            //j-weight matrix col
            //k-weight matrix row
            float a=MAT_AT(n.a[l],0,j);
            float da=MAT_AT(g.a[l],0,j);
            MAT_AT(g.b[l-1],0,j)+=2*da*a*(1-a);
            for(size_t k=0;k<n.a[l-1].cols;k++)
            {
                float pa=MAT_AT(n.a[l-1],0,k);
                float w=MAT_AT(n.w[l-1],k,j);
                MAT_AT(g.w[l-1],k,j)+=2*da*a*(1-a)*pa;
                MAT_AT(g.a[l-1],0,k)+=2*da*a*(1-a)*w;
            }
        }
    }
   }
   for(size_t i=0;i<g.count;i++)
   {
    for(size_t j=0;j<g.w[i].rows;j++)
    {
        for(size_t k=0;k<g.w[i].cols;k++)
        {
            MAT_AT(g.w[i],j,k)/=nn;
        }
    }
    for(size_t j=0;j<g.w[i].rows;j++)
    {
        for(size_t k=0;k<g.w[i].cols;k++)
        {
            MAT_AT(g.b[i],j,k)/=nn;
        }
    }
   }
}
#endif
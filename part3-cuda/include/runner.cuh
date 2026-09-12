#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
// Deterministic CPU double references for every output on small cases;
// stratified plus random checks on large cases. Full cuBLAS checking is added in benchmark.cpp.
int main(int argc,char**argv){
 int m=argc>1?atoi(argv[1]):128,n=argc>2?atoi(argv[2]):128,k=argc>3?atoi(argv[3]):64;
 if(m%128||n%128||k%64||m<128||n<128||k<64||(VERSION<=2&&(m!=128||n!=128))||(VERSION==1&&k!=64)){
  std::cerr<<"Unsupported shape for version "<<VERSION<<"\n";return 2;
 }
 CUDA_OK(cudaSetDevice(0)); cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,0));
 std::vector<H>a(size_t(m)*k),b(size_t(n)*k),d(size_t(m)*n);
 std::mt19937 gen(20260912);std::uniform_real_distribution<float> dist(-1,1);
 for(auto&v:a)v=H(dist(gen));for(auto&v:b)v=H(dist(gen));
 Problem p{m,n,k,prop.multiProcessorCount,nullptr,nullptr,nullptr};
 CUDA_OK(cudaMalloc(&p.a,a.size()*2));CUDA_OK(cudaMalloc(&p.b,b.size()*2));CUDA_OK(cudaMalloc(&p.d,d.size()*2));
 CUDA_OK(cudaMemcpy(p.a,a.data(),a.size()*2,cudaMemcpyHostToDevice));
 CUDA_OK(cudaMemcpy(p.b,b.data(),b.size()*2,cudaMemcpyHostToDevice));
 CUDA_OK(cudaMemset(p.d,0xff,d.size()*2));
 launch(p);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
 CUDA_OK(cudaMemcpy(d.data(),p.d,d.size()*2,cudaMemcpyDeviceToHost));
 double maxerr=0;size_t checks=d.size()<=65536?d.size():4096;
 for(auto v:d)if(!std::isfinite(float(v))){std::cerr<<"nonfinite output\n";return 3;}
 for(size_t i=0;i<checks;++i){
  size_t idx=checks==d.size()?i:(i<1024?i*(d.size()-1)/1023:gen()%d.size());
  int r=idx/n,c=idx%n;double ref=0;
  for(int z=0;z<k;++z)ref+=double(float(a[size_t(r)*k+z]))*float(b[size_t(c)*k+z]);
  double err=std::abs(float(d[idx])-ref);maxerr=std::max(maxerr,err);
  if(err>0.005+0.002*std::abs(ref)){std::cerr<<"FAIL at "<<r<<","<<c<<" got "<<float(d[idx])<<" ref "<<ref<<"\n";return 4;}
 }
 for(int i=0;i<10;++i)launch(p);CUDA_OK(cudaDeviceSynchronize());
 cudaEvent_t s,e;CUDA_OK(cudaEventCreate(&s));CUDA_OK(cudaEventCreate(&e));
 std::vector<float> times;
 for(int sample=0;sample<9;++sample){
  CUDA_OK(cudaEventRecord(s));for(int i=0;i<20;++i)launch(p);CUDA_OK(cudaEventRecord(e));CUDA_OK(cudaEventSynchronize(e));
  float ms;CUDA_OK(cudaEventElapsedTime(&ms,s,e));times.push_back(ms/20);
 }
 std::sort(times.begin(),times.end());float ms=times[4];
 std::cout<<std::setprecision(9)<<"{\"version\":"<<VERSION<<",\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k
 <<",\"correct\":true,\"checked\":"<<checks<<",\"max_abs\":"<<maxerr<<",\"median_ms\":"<<ms
 <<",\"tflops\":"<<2.0*m*n*k/(ms*1e9)<<",\"gpu\":\""<<prop.name<<"\",\"sms\":"<<prop.multiProcessorCount<<"}\n";
 CUDA_OK(cudaFree(p.a));CUDA_OK(cudaFree(p.b));CUDA_OK(cudaFree(p.d));
}

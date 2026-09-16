// CuTe coordinate diagnostic. One GPU thread, no MMA/TMEM data instructions.
// Checks identity tensors only; does not benchmark the GEMM kernels.
#include "common.cuh"
using namespace cute;
template<int G> __device__ int inspect(){
 using Op=std::conditional_t<G==1,SM100_MMA_F16BF16_SS<H,H,float,128,128,UMMA::Major::K,UMMA::Major::K>,SM100_MMA_F16BF16_2x1SM_SS<H,H,float,256,256,UMMA::Major::K,UMMA::Major::K>>;
 using MM=decltype(make_tiled_mma(Op{}));using TileShape=Shape<Int<128*G>,Int<128*G>,_64>;
 auto a=make_identity_tensor(Shape<Int<1024>,Int<192>>{});auto b=a;
 auto d=make_identity_tensor(Shape<Int<1024>,Int<1024>>{});
 auto dt=make_tensor(make_gmem_ptr((H*)nullptr),Layout<Shape<Int<1024>,Int<1024>>,Stride<Int<1024>,_1>>{});
 auto coord=make_coord(1,1,_);
 auto ga=local_tile(a,TileShape{},coord,Step<_1,X,_1>{});auto gb=local_tile(b,TileShape{},coord,Step<X,_1,_1>{});
 auto gd=local_tile(d,TileShape{},coord,Step<_1,_1,X>{});auto gdt=local_tile(dt,TileShape{},coord,Step<_1,_1,X>{});
 int errors=0;
 for(int peer=0;peer<G;peer++){
  auto cta=MM{}.get_slice(peer);auto pa=cta.partition_A(ga),pb=cta.partition_B(gb);auto pd=cta.partition_C(gd);
  printf("G=%d peer=%d pa shape=",G,peer);print(shape(pa));printf(" pb shape=");print(shape(pb));printf(" pd shape=");print(shape(pd));printf("\nA first=");print(pa(make_coord(0,0),0,0,0));printf(" B first=");print(pb(make_coord(0,0),0,0,0));printf(" D first=");print(pd(make_coord(0,0),0,0));printf("\n");
  auto acc=cta.make_fragment_C(cta.partition_C(gdt));
  auto ae=zipped_divide(acc,make_tile(Shape<_128,_64>{}));auto di=zipped_divide(pd,make_tile(Shape<_128,_64>{}));
  auto cp1=make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{},ae(_,Int<0>{}));
  auto cp32=make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{},ae(_,Int<0>{}));
  auto sd_id=group_modes<0,2>(make_identity_tensor(Shape<_128,_64>{}));
  for(int t=0;t<128;t++){
   auto r1=cp1.get_slice(t).partition_D(di(_,Int<0>{}));auto r32=cp32.get_slice(t).partition_D(di(_,Int<0>{}));
   auto s1=cp1.get_slice(t).partition_D(sd_id);auto s32=cp32.get_slice(t).partition_D(sd_id);
   errors+=int(size(r1))!=64 || int(size(r32))!=64;
   for(int j=0;j<64;j++){
    errors+=int(get<0>(r1(j)))!=128*G+peer*128+t || int(get<1>(r1(j)))!=128*G+j;
    errors+=int(get<0>(r32(j)))!=128*G+peer*128+t || int(get<1>(r32(j)))!=128*G+j;
    errors+=int(get<0>(s1(j)))!=t || int(get<1>(s1(j)))!=j;
    errors+=int(get<0>(s32(j)))!=t || int(get<1>(s32(j)))!=j;
   }
   if(t==3){printf("thread3 narrow regs shape=");print(shape(r1));printf(" wide regs shape=");print(shape(r32));printf(" SMEM dst narrow=");print(shape(s1));printf(" wide=");print(shape(s32));printf(" first=");print(r32(0));printf(" last=");print(r32(63));printf("\n");}
  }
 }
 printf("G=%d layout checks errors=%d\n",G,errors);return errors;
}
__global__ void run_probe(int *err){*err=inspect<1>()+inspect<2>();}
int main(){int *p,e;CUDA_OK(cudaMalloc(&p,sizeof(int)));run_probe<<<1,1>>>(p);CUDA_OK(cudaDeviceSynchronize());CUDA_OK(cudaMemcpy(&e,p,sizeof(int),cudaMemcpyDeviceToHost));CUDA_OK(cudaFree(p));return e?1:0;}

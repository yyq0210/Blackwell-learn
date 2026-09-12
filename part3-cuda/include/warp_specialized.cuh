#pragma once
#include "tma.cuh"
namespace study {
template<int G,int C,int Stages>struct WSConfig{
 static constexpr int BM=128*G,BN=128*G,Threads=128*(C+1),Cols=BN*C;
 using Op=std::conditional_t<G==1,SM100_MMA_F16BF16_SS<H,H,float,128,128,UMMA::Major::K,UMMA::Major::K>,SM100_MMA_F16BF16_2x1SM_SS<H,H,float,256,256,UMMA::Major::K,UMMA::Major::K>>;
 using MM=decltype(make_tiled_mma(Op{}));using T=Shape<Int<BM>,Int<BN>,_64>;
 using AShape=decltype(partition_shape_A(MM{},Shape<Int<BM>,_64>{}));
 using BShape=decltype(partition_shape_B(MM{},Shape<Int<BN>,_64>{}));
 using AL=decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<H>{},AShape{}));
 using BL=decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<H>{},BShape{}));
 struct Storage{
  alignas(128) ArrayEngine<H,cosize_v<AL>> a[Stages][C];
  alignas(128) ArrayEngine<H,cosize_v<BL>> b[Stages];
  alignas(128) ArrayEngine<H,cosize_v<LD>> d[C];
  alignas(16) uint64_t full[Stages],empty[Stages],acc_full[C],acc_empty[C];
  alignas(16) uint32_t tmem;
 };
};
template<int G> CUTE_DEVICE void mma_arrive(uint64_t*bar){
 if constexpr(G==1)cutlass::arch::umma_arrive(bar);
 else cutlass::arch::umma_arrive_multicast_2x1SM(bar,3);
}
template<int G,int C,int S,class TA,class TB,class TD,class CA,class CB,class CD>
__global__ void ws_kernel(TA a,TB b,TD d,CUTE_GRID_CONSTANT CA const ca,CUTE_GRID_CONSTANT CB const cb,CUTE_GRID_CONSTANT CD const cd,int mt,int nt){
 using CF=WSConfig<G,C,S>;using MM=typename CF::MM;using Tile=typename CF::T;
 using AL=typename CF::AL;using BL=typename CF::BL;
 extern __shared__ char buf[];auto&s=*reinterpret_cast<typename CF::Storage*>(buf);
 int warp=threadIdx.x/32,peer=blockIdx.x%G,cluster=blockIdx.x/G,cluster_count=gridDim.x/G;
 bool leader=peer==0,elected=elect_one_sync();
 MM mma;auto cta=mma.get_slice(peer);
 using Alloc=std::conditional_t<G==1,TMEM::Allocator1Sm,TMEM::Allocator2Sm>;Alloc alloc;
 if(warp==0)alloc.allocate(CF::Cols,&s.tmem);
 if(threadIdx.x==0){
  for(int i=0;i<S;++i){initialize_barrier(s.full[i],1);initialize_barrier(s.empty[i],C);}
  for(int i=0;i<C;++i){initialize_barrier(s.acc_full[i],1);initialize_barrier(s.acc_empty[i],128*G);}
 }
 if constexpr(G==2)cluster_sync();else __syncthreads();
 // Producer warp: fills a ring. A stage is reusable only after MMA releases it.
 if(warp==4*C+3){
  int iteration=0;
  for(int ti=cluster;ti<mt*nt;ti+=cluster_count){
   int bm,bn;grouped_tile(ti,mt,nt,bm,bn);
   for(int kt=0;kt<size<1>(a)/64;++kt,++iteration){
    int st=iteration%S;
    if(iteration>=S)wait_barrier(s.empty[st],((iteration/S)-1)&1);
    if(elected){
     if(leader)set_barrier_transaction_bytes(s.full[st],G*(C*cosize_v<AL>+cosize_v<BL>)*sizeof(H));
     CUTE_UNROLL
     for(int c=0;c<C;++c){
      auto ga=local_tile(a,Tile{},make_coord(bm*C+c,bn,_),Step<_1,X,_1>{});auto pa=cta.partition_A(ga);
      auto sa=make_tensor(make_smem_ptr(s.a[st][c].begin()),AL{});
      auto [ag,as]=tma_partition(ca,_0{},Layout<_1>{},group_modes<0,3>(sa),group_modes<0,3>(pa));
      copy(ca.with(s.full[st]),ag(_,kt),as);
     }
     auto gb=local_tile(b,Tile{},make_coord(bm*C,bn,_),Step<X,_1,_1>{});auto pb=cta.partition_B(gb);
     auto sb=make_tensor(make_smem_ptr(s.b[st].begin()),BL{});
     auto [bg,bs]=tma_partition(cb,_0{},Layout<_1>{},group_modes<0,3>(sb),group_modes<0,3>(pb));
     copy(cb.with(s.full[st]),bg(_,kt),bs);
    }
   }
  }
 }
 // Exactly C independent MMA issue warps, matching the book's consumer definition.
 if(warp>=4*C&&warp<4*C+C&&leader){
  int c=warp-4*C;
  int iteration=0,round=0;
  for(int ti=cluster;ti<mt*nt;ti+=cluster_count,++round){
   int bm,bn;grouped_tile(ti,mt,nt,bm,bn);
   auto gd=local_tile(d,Tile{},make_coord(bm*C+c,bn,_),Step<_1,_1,X>{});auto pd=cta.partition_C(gd);
   auto acc=cta.make_fragment_C(pd);acc.data()=s.tmem+c*CF::BN;
   if(round)wait_barrier(s.acc_empty[c],(round-1)&1);
   mma.accumulate_=UMMA::ScaleOut::Zero;
   for(int kt=0;kt<size<1>(a)/64;++kt,++iteration){
    int st=iteration%S;wait_barrier(s.full[st],(iteration/S)&1);
    auto sb=make_tensor(make_smem_ptr(s.b[st].begin()),BL{});auto rb=cta.make_fragment_B(sb);
    auto sa=make_tensor(make_smem_ptr(s.a[st][c].begin()),AL{});auto ra=cta.make_fragment_A(sa);
    CUTE_UNROLL
    for(int kb=0;kb<size<2>(ra);++kb){gemm(mma,ra(_,_,kb),rb(_,_,kb),acc);mma.accumulate_=UMMA::ScaleOut::One;}
    // Every consumer signals once: producer expects C completions before reusing A/B.
    mma_arrive<G>(&s.empty[st]);
   }
   mma_arrive<G>(&s.acc_full[c]);
  }
 }
 // Each consumer warpgroup writes one accumulator tile through its own SMEM buffer.
 if(threadIdx.x<128*C){
  int c=threadIdx.x/128,tid=threadIdx.x%128,round=0;
  for(int ti=cluster;ti<mt*nt;ti+=cluster_count,++round){
   int bm,bn;grouped_tile(ti,mt,nt,bm,bn);
   auto gd=local_tile(d,Tile{},make_coord(bm*C+c,bn,_),Step<_1,_1,X>{});auto pd=cta.partition_C(gd);
   auto acc=cta.make_fragment_C(pd);acc.data()=s.tmem+c*CF::BN;
   wait_barrier(s.acc_full[c],round&1);
   auto ae=zipped_divide(acc,make_tile(Shape<_128,_64>{}));auto de=zipped_divide(pd,make_tile(Shape<_128,_64>{}));
   auto sd=make_tensor(make_smem_ptr(s.d[c].begin()),LD{});auto [dg,ds]=tma_partition(cd,sd,de);
   #if defined(STUDY_WIDE_TMEM)
   auto cp=make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{},ae(_,_0{}));
#else
   auto cp=make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{},ae(_,_0{}));
#endif
   auto th=cp.get_slice(tid);
   auto src=th.partition_S(ae);auto dst=th.partition_D(sd);auto rf=make_tensor<float>(shape(dst));auto rh=make_fragment_like(dst);
   for(int ei=0;ei<size<2>(src);++ei){
    copy(cp,src(_,_,ei),rf);
    cutlass::arch::fence_view_async_tmem_load();
    CUTE_UNROLL
    for(int j=0;j<size(rh);++j)rh(j)=H(rf(j));
    copy(rh,dst);tma_store_fence();cutlass::arch::NamedBarrier::sync(128,c);
    if(tid==0){copy(cd,ds,dg(_,ei));tma_store_arrive();tma_store_wait<0>();}
    cutlass::arch::NamedBarrier::sync(128,c);
   }
   if constexpr(G==2)cutlass::arch::ClusterBarrier::arrive(&s.acc_empty[c],0,1);
   else cutlass::arch::ClusterBarrier::arrive(&s.acc_empty[c]);
  }
 }
 if constexpr(G==2)cluster_sync();else __syncthreads();
 if(warp==0){alloc.release_allocation_lock();alloc.free(s.tmem,CF::Cols);}
}
template<int G,int C,int S>void launch_ws(Problem p){
 using CF=WSConfig<G,C,S>;using MM=typename CF::MM;using Tile=typename CF::T;
 auto a=make_tensor(make_gmem_ptr(p.a),make_layout(make_shape(p.m,p.k),make_stride(p.k,_1{})));
 auto b=make_tensor(make_gmem_ptr(p.b),make_layout(make_shape(p.n,p.k),make_stride(p.k,_1{})));
 auto d=make_tensor(make_gmem_ptr(p.d),make_layout(make_shape(p.m,p.n),make_stride(p.n,_1{})));
 auto cluster_layout=tiled_divide(make_layout(Shape<Int<G>,_1,_1>{}),make_tile(typename MM::AtomThrID{}));
 using CopyOp=std::conditional_t<G==1,SM90_TMA_LOAD,SM100_TMA_2SM_LOAD>;
 auto ca=make_tma_atom_A_sm100(CopyOp{},a,typename CF::AL{},Tile{},MM{},cluster_layout);
 auto cb=make_tma_atom_B_sm100(CopyOp{},b,typename CF::BL{},Tile{},MM{},cluster_layout);
 auto cd=make_tma_atom(SM90_TMA_STORE{},d,LD{},Shape<_128,_64>{});
 auto ta=ca.get_tma_tensor(shape(a));auto tb=cb.get_tma_tensor(shape(b));auto td=cd.get_tma_tensor(shape(d));
 auto fn=ws_kernel<G,C,S,decltype(ta),decltype(tb),decltype(td),decltype(ca),decltype(cb),decltype(cd)>;
 static bool configured=false;if(!configured){CUDA_OK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename CF::Storage)));configured=true;}
 int mt=p.m/(CF::BM*C),nt=p.n/CF::BN;int clusters=std::min(mt*nt,p.sms/G);
 if constexpr(G==1)fn<<<clusters,CF::Threads,sizeof(typename CF::Storage)>>>(ta,tb,td,ca,cb,cd,mt,nt);
 else{
  cudaLaunchConfig_t config{};config.gridDim=dim3(clusters*G);config.blockDim=dim3(CF::Threads);config.dynamicSmemBytes=sizeof(typename CF::Storage);config.stream=cudaStreamPerThread;
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={G,1,1};config.attrs=&attr;config.numAttrs=1;
  CUDA_OK(cudaLaunchKernelEx(&config,fn,ta,tb,td,ca,cb,cd,mt,nt));
 }
}
}

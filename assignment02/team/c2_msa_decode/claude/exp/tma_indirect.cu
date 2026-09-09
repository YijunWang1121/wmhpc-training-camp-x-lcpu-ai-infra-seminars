// Discussion point 3 experiment: can TMA express topk_idx -> block_table -> page indirection?
// Three ways to move one (page, kv_head) block (128 pos x 256 [K|V] bf16 = 64 KiB) into smem:
//   (A) cp.async.bulk.tensor.4d  with a tensor map over kv_cache [pages][kvh][128][256], page coord = runtime value
//   (B) cp.async.bulk (1-D)      with a runtime global address (no tensor map at all)
//   (C) plain ld.global by all threads (baseline)
// Each CTA: blk = topk_idx[kh][t][slot]; page = block_table[req][blk]; load; checksum vs. reference.
// Also times 16 back-to-back loads per CTA (pipelined issue vs. serial) to show latency hiding.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
constexpr int PAGE=128, HD=128, ROW=2*HD, KVH=4, TOPK=16;
constexpr uint32_t BLK_BYTES = PAGE*ROW*2;  // 65536

__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt){ asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt)); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes)); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase){
  asm volatile("{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @p bra DONE;\n bra LAB_WAIT;\n DONE:\n}" :: "r"(smem_u32(b)), "r"(phase)); }
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0,int c1,int c2,int c3){
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4,%5,%6}], [%2];"
    :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0),"r"(c1),"r"(c2),"r"(c3) : "memory"); }
__device__ __forceinline__ void bulk_load_1d(void* dst, const void* src, uint32_t bytes, uint64_t* bar){
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
    :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory"); }

struct Args { const __nv_bfloat16* kv; const int32_t* topk; const int32_t* bt; int bt_stride; int total_q; float* out_sum; long long* cycles; int nload; };

// mode 0: TMA 4D map, 1: 1-D bulk, 2: plain loads. grid = (total_q*KVH). Each CTA loads `nload` blocks (slots 0..nload-1).
template<int MODE>
__global__ void __launch_bounds__(128) k_load(const __grid_constant__ CUtensorMap map, Args a){
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ __align__(8) uint64_t bar[TOPK];
  const int t = blockIdx.x % a.total_q, kh = blockIdx.x / a.total_q;
  const int req = t;  // dql = 1
  if(threadIdx.x==0){ for(int i=0;i<a.nload;i++) mbar_init(&bar[i],1); asm volatile("fence.proxy.async.shared::cta;"); }
  __syncthreads();
  long long t0 = clock64();
  // two-level indirection: topk slot -> logical block -> physical page (scalar loads by thread 0)
  if(MODE!=2){
    if(threadIdx.x==0){
      for(int i=0;i<a.nload;i++){
        int blk = a.topk[((size_t)kh*a.total_q + t)*TOPK + i];
        int page = a.bt[(size_t)req*a.bt_stride + blk];
        unsigned char* dst = smem_raw + (size_t)i*BLK_BYTES;
        mbar_expect_tx(&bar[i], BLK_BYTES);
        if(MODE==0) tma_load_4d(dst, &map, &bar[i], 0, 0, kh, page);
        else bulk_load_1d(dst, a.kv + ((size_t)page*KVH + kh)*PAGE*ROW, BLK_BYTES, &bar[i]);
      }
    }
    for(int i=0;i<a.nload;i++) mbar_wait(&bar[i], 0);
  } else {
    for(int i=0;i<a.nload;i++){
      int blk = a.topk[((size_t)kh*a.total_q + t)*TOPK + i];
      int page = a.bt[(size_t)req*a.bt_stride + blk];
      const int4* src = (const int4*)(a.kv + ((size_t)page*KVH + kh)*PAGE*ROW);
      int4* dst = (int4*)(smem_raw + (size_t)i*BLK_BYTES);
      for(int j=threadIdx.x; j<BLK_BYTES/16; j+=blockDim.x) dst[j] = src[j];
    }
    __syncthreads();
  }
  long long t1 = clock64();
  // checksum all loaded bytes (as bf16 -> float)
  float s=0.f; const __nv_bfloat16* sm=(const __nv_bfloat16*)smem_raw;
  for(int j=threadIdx.x; j<a.nload*PAGE*ROW; j+=blockDim.x) s += __bfloat162float(sm[j]);
  __shared__ float red[128]; red[threadIdx.x]=s; __syncthreads();
  if(threadIdx.x==0){ float tot=0; for(int i=0;i<128;i++) tot+=red[i]; a.out_sum[blockIdx.x]=tot; a.cycles[blockIdx.x]=t1-t0; }
}

int main(int argc, char** argv){
  int total_q = argc>1 ? atoi(argv[1]) : 4;   // batch
  int nload   = argc>2 ? atoi(argv[2]) : 16;  // blocks per CTA
  const int nblocks_per_req = 64;              // seq = 8192
  const int npages = total_q*nblocks_per_req;
  std::vector<__nv_bfloat16> kv((size_t)npages*KVH*PAGE*ROW);
  uint32_t rng=12345; for(auto& x: kv){ rng=rng*1664525u+1013904223u; x=__float2bfloat16(((rng>>8)&0xff)/128.f-1.f); }
  // shuffled block table
  std::vector<int32_t> perm(npages); for(int i=0;i<npages;i++) perm[i]=i;
  for(int i=npages-1;i>0;i--){ rng=rng*1664525u+1013904223u; std::swap(perm[i], perm[(rng>>8)%(i+1)]); }
  std::vector<int32_t> bt((size_t)total_q*nblocks_per_req); for(int r=0;r<total_q;r++) for(int b=0;b<nblocks_per_req;b++) bt[r*nblocks_per_req+b]=perm[r*nblocks_per_req+b];
  std::vector<int32_t> topk((size_t)KVH*total_q*TOPK);
  for(int kh=0;kh<KVH;kh++) for(int t=0;t<total_q;t++){ std::vector<int> c(nblocks_per_req); for(int i=0;i<nblocks_per_req;i++) c[i]=i;
    for(int i=nblocks_per_req-1;i>0;i--){ rng=rng*1664525u+1013904223u; std::swap(c[i], c[(rng>>8)%(i+1)]); }
    for(int i=0;i<TOPK;i++) topk[((size_t)kh*total_q+t)*TOPK+i]=c[i]; }
  // reference checksums on host
  std::vector<float> ref((size_t)total_q*KVH);
  for(int kh=0;kh<KVH;kh++) for(int t=0;t<total_q;t++){ double s=0; for(int i=0;i<nload;i++){ int blk=topk[((size_t)kh*total_q+t)*TOPK+i]; int page=bt[t*nblocks_per_req+blk];
      const __nv_bfloat16* p=&kv[((size_t)page*KVH+kh)*PAGE*ROW]; for(int j=0;j<PAGE*ROW;j++) s+=__bfloat162float(p[j]); } ref[kh*total_q+t]=(float)s; }

  __nv_bfloat16* d_kv; int32_t *d_topk,*d_bt; float* d_sum; long long* d_cyc;
  CK(cudaMalloc(&d_kv, kv.size()*2)); CK(cudaMalloc(&d_topk, topk.size()*4)); CK(cudaMalloc(&d_bt, bt.size()*4));
  CK(cudaMalloc(&d_sum, total_q*KVH*4)); CK(cudaMalloc(&d_cyc, total_q*KVH*8));
  CK(cudaMemcpy(d_kv, kv.data(), kv.size()*2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_topk, topk.data(), topk.size()*4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_bt, bt.data(), bt.size()*4, cudaMemcpyHostToDevice));

  // tensor map over kv_cache viewed as 4D [npages][KVH][PAGE][ROW] (innermost first for the driver API)
  CUtensorMap map; cuuint64_t gdim[4]={ROW, PAGE, KVH, (cuuint64_t)npages}; cuuint64_t gstride[3]={ROW*2, (cuuint64_t)PAGE*ROW*2, (cuuint64_t)KVH*PAGE*ROW*2};
  cuuint32_t box[4]={ROW, PAGE, 1, 1}; cuuint32_t estr[4]={1,1,1,1};
  // NOTE: box inner dim bytes = 512 > 256-byte limit for 128B swizzle; use SWIZZLE_NONE (box inner dim limit is 256 elements w/o swizzle)
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, d_kv, gdim, gstride, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); printf("cuTensorMapEncodeTiled failed: %s\n", s); return 1; }

  Args a{d_kv, d_topk, d_bt, nblocks_per_req, total_q, d_sum, d_cyc, nload};
  size_t smem = (size_t)nload*BLK_BYTES;
  auto run=[&](int mode, const char* name){
    void (*k)(const CUtensorMap, Args) = mode==0?k_load<0>: mode==1?k_load<1>: k_load<2>;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    for(int w=0;w<3;w++) k<<<total_q*KVH,128,smem>>>(map,a);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for(int it=0;it<100;it++) k<<<total_q*KVH,128,smem>>>(map,a); cudaEventRecord(e1); CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    std::vector<float> sum(total_q*KVH); std::vector<long long> cyc(total_q*KVH);
    CK(cudaMemcpy(sum.data(), d_sum, sum.size()*4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(cyc.data(), d_cyc, cyc.size()*8, cudaMemcpyDeviceToHost));
    int bad=0; for(size_t i=0;i<sum.size();i++) if(fabsf(sum[i]-ref[i])>1e-2f*fabsf(ref[i])+1.f) bad++;
    long long cmax=0,csum=0; for(auto c: cyc){ cmax=std::max(cmax,c); csum+=c; }
    printf("%-14s batch=%d blocks/CTA=%d smem=%zuKB : %s  kernel %.2f us/launch, in-CTA load cycles avg %lld max %lld (%.2f us @2.03GHz)\n",
      name,total_q,nload,smem/1024, bad?"MISMATCH":"OK", ms*10.f, csum/(long long)cyc.size(), cmax, cmax/2032.f);
  };
  run(0,"TMA-4D-map"); run(1,"bulk-1D"); run(2,"plain-ld");
  return 0;
}

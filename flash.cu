#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O, int* M, const int dm) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int mask_offset = bx * N * dm;
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    int m_tile_size = Bc * dm;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* Mi = &sram[tile_size * 3];
    float* Mj = &sram[tile_size * 3 + m_tile_size];
    float* S = &sram[tile_size * 3 + m_tile_size * 2];
    // float* mask = &sram[tile_size * 3 + m_tile_size * 2 + Br * Bc];

    for (int j = 0; j < Tc; j++) {
        // load Mj to SRAM
        for (int x = 0; x < dm; x++) {
            Mj[(tx * dm) + x] = M[mask_offset + (m_tile_size * j) + (tx * dm) + x];
        }

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Mi to SRAM
            for (int x = 0; x < dm; x++) {
                Mi[(tx * dm) + x] = M[mask_offset + (m_tile_size * i) + (tx * dm) + x];
            }

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // mask = Mi @ Mj^T, apply mask bias
            for (int y = 0; y < Bc; y++) {
                int masksum = 0;
                for (int x = 0; x < dm; x++) {
                    masksum += Mi[(tx * dm) + x] * Mj[(y * dm) + x];
                }
                printf("(%d)", masksum > 0);
                // mask[(Bc * tx) + y] = masksum > 0 ? 1.0f : 0.0f;
                S[(Bc * tx) + y] += masksum > 0 ? 0.0 : -INFINITY;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask_flag) { // mask_flag: type=int32, shape=(bsz, seqlen, dm)
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; 
    const int Br = 32;
    // assert(Bc == Br);

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int dim_mask = mask_flag.size(2);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    int Qi_size = Br * d; 
    int Kj_size = Bc * d; 
    int Vj_size = Bc * d; 
    int S_size = Br * Bc; 
    int mask_size = Br * Bc; 
    int sub_mask_flag_size_row = Br * dim_mask;
    int sub_mask_flag_size_col = Bc * dim_mask;
    const int sram_size = ((Qi_size + Kj_size + Vj_size + S_size) * sizeof(float)) + ((sub_mask_flag_size_row + sub_mask_flag_size_col) * sizeof(int32_t));
    // const int sram_size = ((Qi_size + Kj_size + Vj_size + S_size + mask_size) * sizeof(float)) + ((sub_mask_flag_size_row + sub_mask_flag_size_col) * sizeof(int32_t));

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>(),
        mask_flag.data_ptr<int32_t>(), dim_mask
    );
    return O;
}
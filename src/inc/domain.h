#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__ inline
#else
#define CUDA_CALLABLE_MEMBER
#endif

// Class for cubic domains
class Domain
{
		// Length of edge in grid units
		int dim;
		int DIM;
		
		// ID of the domain
		int id;
		int id3[3];
		
		// ID of the adjacent domains
		int adj[3];
		
		// Lower limits
		int x0[3];
		
		// Upper limits
		int x1[3];
		
	public:
		CUDA_CALLABLE_MEMBER Domain(int ident, int dimension)
		{
			dim = dimension;
			DIM = dim*2;
			id = ident;
			
			id3[0] = id % 2;
			id3[1] = (id / 2) % 2;
			id3[2] = id / 4;
			
			for(int i=0; i<3; i++)
			{
				x0[i] = id3[i]*dim;
				x1[i] = x0[i] + dim;
			}
			
			int p = 1;
			for(int i=0; i<3; i++)
			{
				if(id3[i] == 0)
					adj[i] = id + p;
				else
					adj[i] = id - p;
				p *= 2;
			}
		}
		
		CUDA_CALLABLE_MEMBER int get_dim()
		{
			return dim;
		}
		
		// Return id of the domain
		CUDA_CALLABLE_MEMBER int get_id(void)
		{
			return id;
		}
		
		// Give the 3d id of the domain
		CUDA_CALLABLE_MEMBER void get_id3(int *out)
		{
			for(int i=0; i<3; i++)
			{
				out[i] = id3[i];
			}
		}
		
		// Gives the IDs of the adjacent cells
		CUDA_CALLABLE_MEMBER void get_adj(int* out)
		{
			for(int i=0; i<3; i++)
			{
				out[i] = adj[i];
			}
		}
		
		CUDA_CALLABLE_MEMBER void get_x0(int* x)
		{
			for(int i=0; i<3; i++)
			{
				x[i] = x0[i];
			}
		}
		
		CUDA_CALLABLE_MEMBER void get_x1(int* x)
		{
			for(int i=0; i<3; i++)
			{
				x[i] = x1[i];
			}
		}
		
		CUDA_CALLABLE_MEMBER int loc(int* I)
		{
			if(	I[0] < 0 || I[0] >= DIM ||
				I[1] < 0 || I[1] >= DIM ||
				I[2] < 0 || I[2] >= DIM )
			{
				return -1;
			}
				
			int i0 = I[0] / dim;
			int i1 = I[1] / dim;
			int i2 = I[2] / dim;
			
			return (4*i2 + 2*i1 + i0);
		}
};

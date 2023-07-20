typedef struct source {
    float x;
    float y;
    float z;
    float gam;
} source;

extern "C" {
	void output(float*, float*, short*, float, int);
	void rad(float*, float*, const source*, float*, float*, Ray*, int*, int*, float, float, double*, float*, Domain, float);
	void ion(float*, float*, float*, float*, float*, const float*, float*, float*, float, Domain);
	void dt_H(double*, float, float*, float*, float*, float*, const float*, const float, const float, Domain);
	void my_abort(int err);
	float frand(void);
}

#define DIMX 		256
#define DIMY 			DIMX
#define DIMZ 			DIMX
#define FREQ_BIN_NUM	4		// Number of frequency bins
#define SPECIES 		2		// Number of species to track
#define TNAUGHT			1.e4	// Initial temperature
#define N_PERIOD		10		// Number of slices between bursts
#define Y_P 			0.2501	// Yp
#define NUM_BUF			4000000	// Number of rays to store in each buffer
#define OMEGA_RAY		3.0		// Rays per cell face area
#define PERIODIC		1		// Type of boundary conditions. 1 for periodic

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

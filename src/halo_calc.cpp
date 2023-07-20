#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "./inc/domain.h"
#include "./inc/ray.h"
#include "arc.h"
#include "./inc/background.h"

// Global variables
//#define SCALE 4.0
//#define sigma_eff_dx 1.0/SCALE
#define BOX_SIZE 10.0	// Length of DIMX side in Mpc
//#define BOX_SIZE 0.015
//#define BOX_SIZE 0.0066 // Test123
//#define BOX_SIZE 0.714286 // Test4
#define HUBBLE 0.70			// h
//#define HUBBLE 1.0			// Test
#define MLIM -14.0			// M_lim
#define SINGLE_SOURCE 0.5251
		
void halomatch(float*, unsigned, float, float, int);
double schechter(double,double, double, double);
double model_Mstar(double, unsigned);
double model_log10phi(double, unsigned);
double model_alpha(double, unsigned);

double time(double redshift){
	double h=0.6711;
	double h0=h*3.246753e-18;
	double omegam=0.3;
	double yrtos=3.15569e7;
	double time = 2.*pow((1.+redshift),-3./2.)/(3.*h0*pow(omegam,0.5));
	time = time/(yrtos*1.e6);
	return time;
}

double redshift(double time){
	double h=0.6711;
	double h0=h*3.246753e-18;
	double omegam=0.3;
	double yrtos=3.15569e7;
	time = time*yrtos*1.e6;
	double redshift = pow((3.*h0*pow(omegam,0.5)*time/2.),-2./3.)-1.;
	return redshift;
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

inline float f_esc(float M)
{
	return 1.0-0.5*tanh(M/1.e9);
}

inline float Lum_S0(float mag)
{
	return 9.426e6*exp(0.4*2.30259*(-25.0-mag));
}

bool sortByGam(const source &lhs, const source &rhs) {return lhs.gam > rhs.gam;}

inline void arrayChange(float*);
inline void arrayChangeInv(float*);
inline void BinGridRead(float*, char*, int);

int main(int argc, char** argv)
{
	string line, word;
	ifstream file("../parameters.txt");
	
	string PATH;
	float fDtConst, fFesc, fMdm, fTfinal, fDtFile;
	int nStepType, nLoopInit, nModel;
	
	if(file.good())
	{
		getline(file, line);
		PATH = line.substr(0, line.find("\t"));
		
		getline(file, line);
		nModel = atoi( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		nStepType = atoi( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fDtConst = atof( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fTfinal = atof( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		nLoopInit = atoi( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fDtFile = atof( line.substr(0, line.find("\t")).c_str() );

		getline(file, line);
		fFesc = atof( line.substr(0, line.find("\t")).c_str() );
		
		getline(file, line);
		fMdm = atof( line.substr(0, line.find("\t")).c_str() );
	}
	else
	{
		printf("Couldn't read input file!\n");
		return 0;
	}
	file.close();
	
	FILE *report;
	report = fopen("../dat/report.dat", "w");
	fprintf(report, "Details of the present simulation run:\n");
	fprintf(report, "Time step:\t%f Myr\n", fDtConst);
	fprintf(report, "Length of sim:\t%e\n", fTfinal);
	fprintf(report, "Output produced every:\t%e Myrs\n", fDtFile);
	fclose(report);
	
	cout << "Loading data from: " << PATH << endl;
	cout << "Model: " << nModel << endl;
	cout << "Time step: " << fDtConst << endl;
	cout << "Length of simulation: " << fTfinal << endl;
	cout << "Starting on output number: " << nLoopInit << endl;
	cout << "Output every: " << fDtFile << " Myr" << endl;
	cout << "Escape fraction: " << fFesc << endl;
	
	// Generic string for storing names
	char name[100];
	
	source *particles;

	// Loop variable declarations
	int counter = 0;
	float fTFile = fDtFile*(1.e0 + 1.e-3);
	int step = 0;
	int iFileNum = 0;
	int nOnLimit;	// Max halos (to balance S0)
	int nOnCount;	// Number on in current MODEL
	int nOnFinal;	// Final halo count after limits applied
	
	float fT0 = time(30.0-0.01);
	//float fT0 = time(0.0-0.01);
	//float fT0 = time(9.0-0.01);	//Test
	float fT = 0.0;
	float fDt = fDtConst;

	double ndot;
	
	// Load source file scale factors
	sprintf(name, "../outputs_a.txt");
	
	ifstream scalefactors;
	scalefactors.open(name);
	float nfA[94];	
	for(int i=0; i<94; i++)
	{
		scalefactors >> nfA[i];
	}
	scalefactors.close();
	
	int nDuty = 0;
	float fDuty = 1.0;

	float flux_adjustment = 1.0;
	
	float fTslice = time(1.0 / nfA[iFileNum] - 1.0);

	int iFileNum_init = 0;
	
	for(iFileNum=iFileNum_init; iFileNum<90; iFileNum++)
	{
		float fA = 1.0/(1.0+redshift(fT+fT0));

		if(iFileNum != iFileNum_init)
		{
			delete[] particles;
		}
		nOnCount = 0;
		
		// Open the particle files and figure out how many halos are on
		// 400000 is larger than the largest number of halos
		int *niOnHalos;
		source *p_init;
		float *mags;
		//cout << "Allocated room for halo counts" << endl;
		
		nOnLimit = 0;	// Max halos (to balance S0)
		nOnCount = 0;	// Number on in current MODEL
		
		// Run through the files to find memory requirements
		// Also take note of which halos are on
		
		niOnHalos = new int[400000]; // 400000 is larger than the largest halo file
		
		int iFileNumTemp = iFileNum;
		//iFileNumTemp = 997; // Use this for the test file
		//iFileNumTemp = 1; // Use this for the test file
		//sprintf(name, "../slice%03d.dat", iFileNumTemp);
		sprintf(name, "%s/nanoJubilee/Reduced/slice%02d.dat", PATH.c_str(), iFileNumTemp);
		//sprintf(name, "../sources.dat");	//Test4
		
		string line;
		ifstream file(name);
		
		int i=0;
		while (getline(file, line))
		{
			string ID;
			float X, Y, Z, a;
			float mStarC, mStarB;
			int mdm, slice, state;
			istringstream iss(line);
			
			if(!(	iss >> ID >> mdm >>
					X >> Y >> Z >> a >>
					slice >> state >> 
					mStarC >> mStarB))
			{
				break;
			}

			/*if(!(	iss >> ID >> mdm >>
					X >> Y >> Z >> a >>
					slice >> state ))
			{
				break;
			}*/
			
			/*mdm=0;
			slice=0;
			state=0;
			if(!(iss >> X >> Y >> Z >> a))
				break;*///Test4
			
			float halo_mass = 23936.0*mdm;
			
			//cout << slice << "\t" << state << endl;
			
			// Condition for a halo being on
			// Pop III star conditions would be set here
			niOnHalos[i] = 0;

			if(state==0)	//has merged
			{
				//printf("mass = %e, fMdm = %e\n", mass, fMdm);
				if(nModel == 0)
				{
					niOnHalos[i] = 1;
					nOnCount++;
				}
				
				if( (halo_mass > fMdm && mStarB < 1.0) || // Massive enough and no stars
					((iFileNum - slice)%10 == 0 && mStarB > 1.0) ) // Bursting again
				{
					nOnLimit++;	// Number of bursty halos sets limit
				
					if(nModel == 1)
					{
						niOnHalos[i] = 1;
						printf("Turning on halo %d\n", i);
						nOnCount++;
					}
				}
			}
			
			i++;
		}
		
		// Run through the file again to copy relevant halo data
		p_init = new source[nOnCount];
		printf("Total Halos = %d\tnOnLimit = %d\tnOnCount = %d\n", i, nOnLimit, nOnCount);
		
		file.clear();
		file.seekg(0, ios::beg);
		
		i = 0;
		int j = 0;
		int min=100, max=0;
		while (getline(file, line))
		{
			string ID;
			float X, Y, Z, a;
			float temp;
			int mdm, slice, state;
			istringstream iss(line);
			
			// Read the line into the relevant variables
			/*if(!(iss >> ID >> mdm >> X >> Y >> Z >> a >> slice >> state >> temp >> temp))
				break;*/
			if(!(iss >> ID >> mdm >> X >> Y >> Z >> a >> slice >> state))
				break;
			
			//cout << ID << "\t" << mdm << endl;
			//float mass = 23936.0*mdm;
			if(niOnHalos[i] == 1)
			{
				p_init[j].gam = (float) mdm;	// Mass in #dm particles
				/*p_init[j].x = (X/(BOX_SIZE*1.e3*HUBBLE))*DIMX*0.0066/10.0;
				p_init[j].y = (Y/(BOX_SIZE*1.e3*HUBBLE))*DIMY*0.0066/10.0;
				p_init[j].z = (Z/(BOX_SIZE*1.e3*HUBBLE))*DIMZ*0.0066/10.0;*/
				p_init[j].x = (X/(BOX_SIZE*1.e3*HUBBLE))*DIMX;
				p_init[j].y = (Y/(BOX_SIZE*1.e3*HUBBLE))*DIMY;
				p_init[j].z = (Z/(BOX_SIZE*1.e3*HUBBLE))*DIMZ;
				j++;
			}
			
			/*if(!(iss >> X >> Y >> Z >> a))
				break;
			
			//cout << ID << "\t" << mdm << endl;
			if(niOnHalos[i] == 1)
			{
				p_init[j].gam = a;
				p_init[j].x = X;
				p_init[j].y = Y;
				p_init[j].z = Z;
				
				j++;
			}*///Test
			
			i++;
			
		}
		
		delete[] niOnHalos;
		file.close();
		
		// Put the halos in order of decreasing mass
		sort(p_init, p_init + nOnCount, sortByGam);
		//cout << "p_init[0] = " << p_init[0].gam << endl;
		//cout << "p_init[nOnLimit] = " << p_init[nOnLimit].gam << endl;
		
		// Calculate luminosities and how many halos are above MLIM
		//nOnLimit = nOnCount; //Test4
		mags = new float[nOnLimit];
		
		halomatch(mags, nOnLimit, nfA[iFileNum], pow(BOX_SIZE,3), 0);	// HUBBLE?
		
		nOnFinal = 0;
		
		for (int i=0; i<nOnLimit; i++)
		{
			//if(true)//Test
			if( mags[i] <= MLIM )
			{
				nOnFinal++;
				
				float dim2 = DIMX/2.0;
				int ix = (int) p_init[i].x/dim2;
				int iy = (int) p_init[i].y/dim2;
				int iz = (int) p_init[i].z/dim2;
				
				int ind = 4*iz + 2*iy + ix;
				//nOnLoc[ind]++;
			}
		}
		
		particles = new source[nOnFinal];
		ndot = 0; // Total photon count for current slice
		
		// Load the relevant halos into the particle array
		float *mass;
		mass = new float[nOnFinal];
		
		float fJet = 1.0;
		for (int i=0; i<nOnFinal; i++)
		{
			float dim2 = DIMX/2.0;
			int ix = (int) p_init[i].x/dim2;
			int iy = (int) p_init[i].y/dim2;
			int iz = (int) p_init[i].z/dim2;
			int ind = 4*iz + 2*iy + ix;
			
			int j = i;
			//printf("%d\t%d\t%d\t%d\n", ind, nOnDisp[ind], (nOnLocTemp[ind]-1), j);
			
			particles[j].x = p_init[i].x;
			particles[j].y = p_init[i].y;
			particles[j].z = p_init[i].z;
			particles[j].gam = p_init[i].gam;
			
			float S0 = Lum_S0(mags[i]);
			
			mass[j] = p_init[i].gam*23936.0;
			
			particles[j].gam = fJet*fFesc*S0/fDuty;	// CHECK
			//particles[j].gam = 1.e3*fFesc*SINGLE_SOURCE;	// TEST
			//particles[j].gam = p_init[i].gam*1050.0;	// TEST
			
			// ndi is the photons per particle
			float ndi = particles[j].gam;
			ndi /= 2.43e-7*pow((float) BOX_SIZE,3.0)*3.086e24*fJet;
			ndot += ndi;	//TestCHECKXXX
			
			//printf(	"Halo %d added! S0, (n_x, n_y, nz) = %f (%f, %f, %f)\n",
			//			ind, particles[j].gam,particles[j].x, particles[j].y, particles[j].z);
		}
		
		char mlname[40];
		sprintf(mlname, "../dat/photons/slist%02d.dat", iFileNum);

		FILE *ML;
		ML = fopen(mlname, "w");
		
		for(int i=0; i<nOnFinal; i++)
		{
			fprintf(ML, "%e\t%e\t", particles[i].x, particles[i].y);
			fprintf(ML, "%e\t%e\t", particles[i].z, particles[i].gam/fJet/fFesc);
			fprintf(ML, "%e\n", mass[i]);
		}
		
		fclose(ML);
		
		delete[] p_init;
		
		delete[] mags;
		delete[] mass;
		
		// Adjustment
		sprintf(mlname, "../dat/photons/slist%02d.dat", iFileNum+1);
		
		ifstream next_slist(mlname);
		
		float S_temp=0, S_now=0, S_next=0;

		for(int i=0; i<nOnFinal; i++)
		{
			S_now += particles[i].gam/fJet/fFesc;
		}

		int i_next = 0;
		for(int i=0; getline(next_slist, line); i++)
		{
			istringstream iss(line);
			iss >> S_temp >> S_temp >> S_temp >> S_temp;

			S_next += S_temp;
			i_next += 1;
		}
		next_slist.close();
		flux_adjustment = S_next/S_now;
		printf("S_now = (%d, %e)\tS_next = (%d,%e)\n", nOnFinal, S_now, i_next, S_next);
		printf("Adjustment factor for slice %d = %e\n\n", iFileNum, flux_adjustment);
		
		nDuty = 0;
	}
			
	return 0;
}

inline void arrayChange(float* array)
{
	int mainIndex=0;
	int subIndex;
	int size = DIMX*DIMX*DIMX;
	int half = DIMX/2;
	
	float* buffer = new float[size];
	memcpy(buffer, array, size*sizeof(float));
	
	for(int k=0; k<DIMX; k++)
	{
		for(int j=0; j<DIMX; j++)
		{
			for(int i=0; i<DIMX; i++)
			{
				int tag = 4*(k/half) + 2*(j/half) + 1*(i/half);
				subIndex = half*half*(k%half);
				subIndex += half*(j%half);
				subIndex += i%half;
				subIndex += tag*half*half*half;
				
				array[subIndex] = buffer[mainIndex];
				
				mainIndex++;
			}
		}
	}
	
	delete[] buffer;
}
inline void arrayChangeInv(float* array)
{
	int mainIndex=0;
	int subIndex;
	int size = DIMX*DIMX*DIMX;
	int half = DIMX/2;
	
	float* buffer = new float[size];
	memcpy(buffer, array, size*sizeof(float));
	
	for(int k=0; k<DIMX; k++)
	{
		for(int j=0; j<DIMX; j++)
		{
			for(int i=0; i<DIMX; i++)
			{
				int tag = 4*(k/half) + 2*(j/half) + 1*(i/half);
				subIndex = half*half*(k%half);
				subIndex += half*(j%half);
				subIndex += i%half;
				subIndex += tag*half*half*half;
				
				array[mainIndex] = buffer[subIndex];
				
				mainIndex++;
			}
		}
	}
	
	delete[] buffer;
}

inline void BinGridRead(float* array, char* filename, int dim)
{
	ifstream ifsFile (filename, ios::in | ios::binary);
	ifsFile.seekg (0, ios::beg);
	
	unsigned long int num = dim*dim*dim;
	
	for (unsigned long int i = 0; i < num; i++)
	{
		float buff;
		
		ifsFile.read((char*)&buff, sizeof(float));
		array[i] = buff;
	}
	
	ifsFile.close();
}

// Returns magnitude of Nth halo
void halomatch(float* L, unsigned n, float a, float V, int mod)
{
	float z = 1.0/a-1.0;
	float lmin = 0, lmax = -25, dl = 0.01;
	float nschechter = 0;
	float ll = lmax;
	
	for(unsigned i=0; i<n; i++)
	{
		float dndl = schechter(ll, model_Mstar(z, mod), model_log10phi(z, mod), model_alpha(z, mod));
		while(nschechter + dl*dndl*10 > (i+1)/V && nschechter < (i+1)/V)
		{
			//cout << "Decreasing step size" << dl << " -> " << dl/10 << endl;
			dl = dl/10;
		}
		while(nschechter < (i+1)/V)
		{
			ll += dl;
			dndl = schechter(ll, model_Mstar(z, mod), model_log10phi(z, mod), model_alpha(z, mod));
			nschechter += dl*dndl;
		}
		
		//cout << "N = " << nschechter << "\tM = " << ll << endl;
		//cout << "i = " << i << "\tdndnl = " << dndl << endl;
		L[i] = ll;
	}
}

double schechter(	double M,			// Mass
					double Mstar,		// Shechter parameters 
					double log10phi,		// "
					double alpha)		// "
{
	double temp = pow(10.,0.4*(Mstar-M));
	return 0.921034*pow(10.,log10phi)*pow(temp,alpha+1.)*pow(2.718281828,-temp);
}

double model_Mstar(double z, unsigned mod)
{
	if(mod == 1)	{return -20.37 + 0.30*(z - 6.);}
	else if(mod == 2)	{return -20.47 + 0.24*(z - 6.);}
	return -20.42 + 0.27*(z - 6.);
}

double model_log10phi(double z, unsigned mod)
{
	if(mod == 1)	{return -3.05 - 0.09*(z - 6.);}
	else if(mod == 2)	{return -2.97 - 0.05*(z - 6.);}
	return -3.01 - 0.07*(z - 6.);
}

double model_alpha(double z, unsigned mod)
{
	if(mod == 1)	{return -1.80 - 0.04*(z - 6.);}
	else if(mod == 2)	{return -1.88 - 0.08*(z - 6.);}
	return -1.84 - 0.06*(z - 6.);
}

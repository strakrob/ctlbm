#define _dfeq(q1,q2,q3) (rho + (- (T).5*iCsq * (u.x*u.x + u.y*u.y + u.z*u.z) + iCsq*(q1*u.x + q2*u.y + q3*u.z) + (T).5*iCsq*iCsq*(q1*u.x + q2*u.y + q3*u.z)*(q1*u.x + q2*u.y + q3*u.z)))
__device__ T _feq_zzz(T rho, T2 u) {return Wc*_dfeq( 0, 0, 0);}
__device__ T _feq_pzz(T rho, T2 u) {return Ws*_dfeq( 1, 0, 0);}
__device__ T _feq_mzz(T rho, T2 u) {return Ws*_dfeq(-1, 0, 0);}
__device__ T _feq_zpz(T rho, T2 u) {return Ws*_dfeq( 0, 1, 0);}
__device__ T _feq_zmz(T rho, T2 u) {return Ws*_dfeq( 0,-1, 0);}
__device__ T _feq_zzp(T rho, T2 u) {return Ws*_dfeq( 0, 0, 1);}
__device__ T _feq_zzm(T rho, T2 u) {return Ws*_dfeq( 0, 0,-1);}
__device__ T _feq_ppz(T rho, T2 u) {return Wm*_dfeq( 1, 1, 0);}
__device__ T _feq_pmz(T rho, T2 u) {return Wm*_dfeq( 1,-1, 0);}
__device__ T _feq_mpz(T rho, T2 u) {return Wm*_dfeq(-1, 1, 0);}
__device__ T _feq_mmz(T rho, T2 u) {return Wm*_dfeq(-1,-1, 0);}
__device__ T _feq_pzp(T rho, T2 u) {return Wm*_dfeq( 1, 0, 1);}
__device__ T _feq_mzm(T rho, T2 u) {return Wm*_dfeq(-1, 0,-1);}
__device__ T _feq_pzm(T rho, T2 u) {return Wm*_dfeq( 1, 0,-1);}
__device__ T _feq_mzp(T rho, T2 u) {return Wm*_dfeq(-1, 0, 1);}
__device__ T _feq_zpp(T rho, T2 u) {return Wm*_dfeq( 0, 1, 1);}
__device__ T _feq_zpm(T rho, T2 u) {return Wm*_dfeq( 0, 1,-1);}
__device__ T _feq_zmp(T rho, T2 u) {return Wm*_dfeq( 0,-1, 1);}
__device__ T _feq_zmm(T rho, T2 u) {return Wm*_dfeq( 0,-1,-1);}
__device__ T _feq_ppp(T rho, T2 u) {return Wl*_dfeq( 1, 1, 1);}
__device__ T _feq_mmm(T rho, T2 u) {return Wl*_dfeq(-1,-1,-1);}
__device__ T _feq_ppm(T rho, T2 u) {return Wl*_dfeq( 1, 1,-1);}
__device__ T _feq_pmp(T rho, T2 u) {return Wl*_dfeq( 1,-1, 1);}
__device__ T _feq_mpp(T rho, T2 u) {return Wl*_dfeq(-1, 1, 1);}
__device__ T _feq_mpm(T rho, T2 u) {return Wl*_dfeq(-1, 1,-1);}
__device__ T _feq_mmp(T rho, T2 u) {return Wl*_dfeq(-1,-1, 1);}
__device__ T _feq_pmm(T rho, T2 u) {return Wl*_dfeq( 1,-1,-1);}

__global__ void LBMCollideAndPropagate(char *map, T *cd, T *od, T *macro)
{
 int x = threadIdx.x;
 int z = blockIdx.x;
 int y = blockIdx.y;
 int gi,mapgi;

 int xp = (x == LAT_W-1) ? 0 : (x+1);
 int xm = (x == 0) ? (LAT_W-1) : (x-1);
 int yp = (y == LAT_H-1) ? 0 : (y+1);
 int ym = (y == 0) ? (LAT_H-1) : (y-1);
 int zp = (z == LAT_D-1) ? 0 : (z+1);
 int zm = (z == 0) ? (LAT_D-1) : (z-1);

 T fzzz, fpzz, fmzz, fzpz, fzmz, fzzp, fzzm; // Q6
 T fppz, fpmz, fmpz, fmmz;
 T fpzp, fpzm, fmzp, fmzm;
 T fzpp, fzpm, fzmp, fzmm; //Q19
 T fppp, fppm, fpmp, fpmm;
 T fmpp, fmpm, fmmp, fmmm; // Q27
 T rho;
 T2 u;

 fzzz = cd[Fxyz( 0, 0, 0,x ,y ,z )];
 fpzz = cd[Fxyz( 1, 0, 0,xm,y ,z )];
 fmzz = cd[Fxyz(-1, 0, 0,xp,y ,z )];
 fzpz = cd[Fxyz( 0, 1, 0,x ,ym,z )];
 fzmz = cd[Fxyz( 0,-1, 0,x ,yp,z )];
 fzzp = cd[Fxyz( 0, 0, 1,x ,y ,zm)];
 fzzm = cd[Fxyz( 0, 0,-1,x ,y ,zp)];
 fppz = cd[Fxyz( 1, 1, 0,xm,ym,z )];
 fpmz = cd[Fxyz( 1,-1, 0,xm,yp,z )];
 fmpz = cd[Fxyz(-1, 1, 0,xp,ym,z )];
 fmmz = cd[Fxyz(-1,-1, 0,xp,yp,z )];
 fpzp = cd[Fxyz( 1, 0, 1,xm,y ,zm)];
 fpzm = cd[Fxyz( 1, 0,-1,xm,y ,zp)];
 fmzp = cd[Fxyz(-1, 0, 1,xp,y ,zm)];
 fmzm = cd[Fxyz(-1, 0,-1,xp,y ,zp)];
 fzpp = cd[Fxyz( 0, 1, 1,x ,ym,zm)];
 fzpm = cd[Fxyz( 0, 1,-1,x ,ym,zp)];
 fzmp = cd[Fxyz( 0,-1, 1,x ,yp,zm)];
 fzmm = cd[Fxyz( 0,-1,-1,x ,yp,zp)];

 fppp = cd[Fxyz( 1, 1, 1,xm,ym,zm)];
 fppm = cd[Fxyz( 1, 1,-1,xm,ym,zp)];
 fpmp = cd[Fxyz( 1,-1, 1,xm,yp,zm)];
 fpmm = cd[Fxyz( 1,-1,-1,xm,yp,zp)];
 fmpp = cd[Fxyz(-1, 1, 1,xp,ym,zm)];
 fmpm = cd[Fxyz(-1, 1,-1,xp,ym,zp)];
 fmmp = cd[Fxyz(-1,-1, 1,xp,yp,zm)];
 fmmm = cd[Fxyz(-1,-1,-1,xp,yp,zp)];


  if ((mapgi&1) == GEO_FLUID){
    // macroscopic quantities for the current cell
    rho = fzzz+fpzz+fmzz+fzpz+fzmz+fzzp+fzzm+
 		fppz+fpmz+fmpz+fmmz+
		fpzp+fpzm+fmzp+fmzm+
		fzpp+fzpm+fzmp+fzmm+
		fppp+fppm+fpmp+fpmm+
		fmpp+fmpm+fmmp+fmmm; // Q27
	const T rho_inv = (T)1./rho;
    u.x = (fpzz-fmzz+fppz+fpmz-fmpz-fmmz+fpzp+fpzm-fmzp-fmzm+fppp+fppm+fpmp+fpmm-fmpp-fmpm-fmmp-fmmm)*rho_inv;
    u.y = (fzpz-fzmz+fppz-fpmz+fmpz-fmmz+fzpp+fzpm-fzmp-fzmm+fppp+fppm-fpmp-fpmm+fmpp+fmpm-fmmp-fmmm)*rho_inv;
    u.z = (fzzp-fzzm+fpzp-fpzm+fmzp-fmzm+fzpp-fzpm+fzmp-fzmm+fppp-fppm+fpmp-fpmm+fmpp-fmpm+fmmp-fmmm)*rho_inv;
 
		// collision step: BGK

		const T omega = (T)1./(no3*visclb+n1o2);
		fmmm += omega*(_feq_mmm(rho,u) - fmmm);
		fmmz += omega*(_feq_mmz(rho,u) - fmmz);
		fmmp += omega*(_feq_mmp(rho,u) - fmmp);
		fmzm += omega*(_feq_mzm(rho,u) - fmzm);
		fmzz += omega*(_feq_mzz(rho,u) - fmzz);
		fmzp += omega*(_feq_mzp(rho,u) - fmzp);
		fmpm += omega*(_feq_mpm(rho,u) - fmpm);
		fmpz += omega*(_feq_mpz(rho,u) - fmpz);
		fmpp += omega*(_feq_mpp(rho,u) - fmpp);
		fzmm += omega*(_feq_zmm(rho,u) - fzmm);
		fzmz += omega*(_feq_zmz(rho,u) - fzmz);
		fzmp += omega*(_feq_zmp(rho,u) - fzmp);
		fzzm += omega*(_feq_zzm(rho,u) - fzzm);
		fzzz += omega*(_feq_zzz(rho,u) - fzzz);
		fzzp += omega*(_feq_zzp(rho,u) - fzzp);
		fzpm += omega*(_feq_zpm(rho,u) - fzpm);
		fzpz += omega*(_feq_zpz(rho,u) - fzpz);
		fzpp += omega*(_feq_zpp(rho,u) - fzpp);
		fpmm += omega*(_feq_pmm(rho,u) - fpmm);
		fpmz += omega*(_feq_pmz(rho,u) - fpmz);
		fpmp += omega*(_feq_pmp(rho,u) - fpmp);
		fpzm += omega*(_feq_pzm(rho,u) - fpzm);
		fpzz += omega*(_feq_pzz(rho,u) - fpzz);
		fpzp += omega*(_feq_pzp(rho,u) - fpzp);
		fppm += omega*(_feq_ppm(rho,u) - fppm);
		fppz += omega*(_feq_ppz(rho,u) - fppz);
		fppp += omega*(_feq_ppp(rho,u) - fppp);
     macro[V(F_RHO,gi)] = rho;
     macro[V(F_VELX,gi)] = u.x;
     macro[V(F_VELY,gi)] = u.y;
     macro[V(F_VELZ,gi)] = u.z;

 	 od[Fxyz( 0, 0, 0,x,y,z)] = fzzz;
	 od[Fxyz( 1, 0, 0,x,y,z)] = fpzz;
	 od[Fxyz(-1, 0, 0,x,y,z)] = fmzz;
	 od[Fxyz( 0, 1, 0,x,y,z)] = fzpz;
	 od[Fxyz( 0,-1, 0,x,y,z)] = fzmz;
	 od[Fxyz( 0, 0, 1,x,y,z)] = fzzp;
	 od[Fxyz( 0, 0,-1,x,y,z)] = fzzm;
	 od[Fxyz( 1, 1, 0,x,y,z)] = fppz;
	 od[Fxyz( 1,-1, 0,x,y,z)] = fpmz;
	 od[Fxyz(-1, 1, 0,x,y,z)] = fmpz;
	 od[Fxyz(-1,-1, 0,x,y,z)] = fmmz;
	 od[Fxyz( 1, 0, 1,x,y,z)] = fpzp;
	 od[Fxyz( 1, 0,-1,x,y,z)] = fpzm;
	 od[Fxyz(-1, 0, 1,x,y,z)] = fmzp;
	 od[Fxyz(-1, 0,-1,x,y,z)] = fmzm;
 	 od[Fxyz( 0, 1, 1,x,y,z)] = fzpp;
	 od[Fxyz( 0, 1,-1,x,y,z)] = fzpm;
	 od[Fxyz( 0,-1, 1,x,y,z)] = fzmp;
	 od[Fxyz( 0,-1,-1,x,y,z)] = fzmm;
                             
	 od[Fxyz( 1, 1, 1,x,y,z)] = fppp;
	 od[Fxyz( 1, 1,-1,x,y,z)] = fppm;
	 od[Fxyz( 1,-1, 1,x,y,z)] = fpmp;
	 od[Fxyz( 1,-1,-1,x,y,z)] = fpmm;
	 od[Fxyz(-1, 1, 1,x,y,z)] = fmpp;
	 od[Fxyz(-1, 1,-1,x,y,z)] = fmpm;
	 od[Fxyz(-1,-1, 1,x,y,z)] = fmmp;
	 od[Fxyz(-1,-1,-1,x,y,z)] = fmmm;

}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>

static double entry(const double* o, const double* d, const double* c, double r) {
    double x0=o[0]-c[0], x1=o[1]-c[1], x2=o[2]-c[2];
    double aa=(d[0]*d[0]+d[1]*d[1])+d[2]*d[2];
    double projection=((x0*d[0]+x1*d[1])+x2*d[2])/aa;
    double p0=x0-projection*d[0], p1=x1-projection*d[1], p2=x2-projection*d[2];
    double disc=r*r-((p0*p0+p1*p1)+p2*p2);
    if(disc<0. || projection>=0.) return INFINITY;
    double near=-projection-std::sqrt(std::max(0.,disc)/aa);
    if(near < -64*std::numeric_limits<double>::epsilon()*r) return INFINITY;
    return std::max(0.,near);
}

static bool box(const double* o,const double* d,const double* lo,const double* hi,double best) {
    double near=0.,far=INFINITY;
    for(int c=0;c<3;++c) {
        if(d[c]==0.) { if(o[c]<lo[c] || o[c]>hi[c]) return false; }
        else {
            double a=(lo[c]-o[c])/d[c], b=(hi[c]-o[c])/d[c];
            near=std::max(near,std::min(a,b)); far=std::min(far,std::max(a,b));
        }
    }
    return far>=near && near<=best && far>=0.;
}

static void reflect(double* o,double* d,const double* c,double r,double distance) {
    double n0=o[0]+distance*d[0]-c[0], n1=o[1]+distance*d[1]-c[1], n2=o[2]+distance*d[2]-c[2];
    double norm=std::sqrt((n0*n0+n1*n1)+n2*n2);
    n0/=norm; n1/=norm; n2/=norm;
    o[0]=c[0]+r*n0; o[1]=c[1]+r*n1; o[2]=c[2]+r*n2;
    double dot=(d[0]*n0+d[1]*n1)+d[2]*n2;
    d[0]=d[0]-2*dot*n0; d[1]=d[1]-2*dot*n1; d[2]=d[2]-2*dot*n2;
    norm=std::sqrt((d[0]*d[0]+d[1]*d[1])+d[2]*d[2]);
    d[0]/=norm; d[1]/=norm; d[2]/=norm;
}

extern "C" __declspec(dllexport) void ehss_trace(
    int64_t nr,int64_t nn,int cap,int threads,int record,
    double* p,double* d,const int64_t* lasts,int64_t* counts,uint8_t* escaped,uint8_t* unresolved,int32_t* history,
    const double* lo,const double* hi,const int64_t* end,const int64_t* leaf,const int64_t* offsets,
    const int64_t* ids,const double* world,const double* radii,int64_t* counters,int* observed_threads) {
    int64_t boxes=0,spheres=0;
    int observed=1;
    #pragma omp parallel num_threads(threads) reduction(+:boxes,spheres)
    {
        #pragma omp single
        observed=omp_get_num_threads();
        #pragma omp for schedule(static)
        for(int64_t ray=0;ray<nr;++ray) {
            double* o=p+3*ray; double* v=d+3*ray;
            int64_t last=lasts[ray];
            if(record && counts[ray]==1) history[ray*cap]=static_cast<int32_t>(last);
            while(true) {
                int64_t atom=-1,node=0; double best=INFINITY;
                while(node<nn) {
                    ++boxes;
                    if(!box(o,v,lo+3*node,hi+3*node,best)) { node=end[node]; continue; }
                    int64_t g=leaf[node];
                    if(g>=0) for(int64_t k=offsets[g];k<offsets[g+1];++k) {
                        int64_t wanted=ids[k]; if(wanted==last) continue;
                        ++spheres; double distance=entry(o,v,world+3*wanted,radii[wanted]);
                        if(distance<best || (distance==best && std::isfinite(distance) && (atom<0 || wanted<atom))) {
                            atom=wanted; best=distance;
                        }
                    }
                    ++node;
                }
                if(atom<0) { escaped[ray]=1; break; }
                if(counts[ray]>=cap) { unresolved[ray]=1; break; }
                reflect(o,v,world+3*atom,radii[atom],best);
                if(record) history[ray*cap+counts[ray]]=static_cast<int32_t>(atom);
                ++counts[ray]; last=atom;
            }
        }
    }
    counters[0]=boxes; counters[1]=spheres; *observed_threads=observed;
}

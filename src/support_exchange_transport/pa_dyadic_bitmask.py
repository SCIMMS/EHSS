"""Exact finite-grid PA union using packed OR and row-wise dyadic prefixes.

Bit addresses identify projection pixels, never atoms. This is a p-adic-like
binary hierarchy, not a new p-adic field algorithm. Prefixes operate within a
row; they are not a 2D Morton quadtree. Merging requires the SAME physical grid.
"""
from dataclasses import dataclass
import math
import numpy as np
from numba import njit

U64=np.uint64

@njit(cache=True)
def low_bits(n):
    if n>=64:return np.uint64(0xffffffffffffffff)
    return (np.uint64(1)<<np.uint64(n))-np.uint64(1)

@njit(cache=True)
def pop64(x):
    x=x-((x>>np.uint64(1))&np.uint64(0x5555555555555555))
    x=(x&np.uint64(0x3333333333333333))+((x>>np.uint64(2))&np.uint64(0x3333333333333333))
    x=(x+(x>>np.uint64(4)))&np.uint64(0x0f0f0f0f0f0f0f0f)
    return int((x*np.uint64(0x0101010101010101))>>np.uint64(56))

@njit(cache=True)
def packed_count(words):
    count=0
    for y in range(words.shape[0]):
        for k in range(words.shape[1]):count+=pop64(words[y,k])
    return count

@njit(cache=True)
def or_span(words,y,lo,hi):
    writes=0
    if hi<=lo:return writes
    a=lo//64;b=(hi-1)//64
    for k in range(a,b+1):
        left=max(lo-k*64,0);right=min(hi-k*64,64)
        words[y,k]|=low_bits(right)^low_bits(left);writes+=1
    return writes

@njit(cache=True)
def packed_intervals(intervals):
    n=intervals.shape[1];words=np.zeros((n,(n+63)//64),np.uint64);writes=0
    for a in range(len(intervals)):
        for y in range(n):writes+=or_span(words,y,intervals[a,y,0],intervals[a,y,1])
    return words,writes

@njit(cache=True)
def row_span(cx,cy,r,lower,spacing,y,n):
    dy=lower[1]+(y+.5)*spacing-cy;rem=r*r-dy*dy
    if rem<0:return 0,0
    half=math.sqrt(rem)
    lo=max(0,min(n,int(math.ceil((cx-half-lower[0])/spacing-.5))))
    hi=max(0,min(n,int(math.floor((cx+half-lower[0])/spacing-.5))+1))
    while lo>0 and (lower[0]+(lo-.5)*spacing-cx)**2<=rem:lo-=1
    while lo<hi and (lower[0]+(lo+.5)*spacing-cx)**2>rem:lo+=1
    while hi<n and (lower[0]+(hi+.5)*spacing-cx)**2<=rem:hi+=1
    while hi>lo and (lower[0]+(hi-.5)*spacing-cx)**2>rem:hi-=1
    return lo,hi

@njit(cache=True)
def row_bounds(cy,r,lower,spacing,n):
    # Expand one row around rounded endpoints, then check exact inequalities.
    lo=max(0,min(n,int(math.ceil((cy-r-lower[1])/spacing-.5))-1))
    hi=max(0,min(n,int(math.floor((cy+r-lower[1])/spacing-.5))+2))
    return lo,hi

@njit(cache=True)
def packed_disks(xy,radii,lower,spacing,n):
    words=np.zeros((n,(n+63)//64),np.uint64);rows=0;writes=0
    for a in range(len(radii)):
        first,stop=row_bounds(xy[a,1],radii[a],lower,spacing,n)
        for y in range(first,stop):
            lo,hi=row_span(xy[a,0],xy[a,1],radii[a],lower,spacing,y,n);rows+=1
            writes+=or_span(words,y,lo,hi)
    return words,rows,writes

@njit(cache=True)
def count_disks(xy,radii,lower,spacing,n):
    """Matched streaming baseline: SAME row selection and interval arithmetic."""
    delta=np.zeros((n,n+1),np.int64);rows=0
    for a in range(len(radii)):
        first,stop=row_bounds(xy[a,1],radii[a],lower,spacing,n)
        for y in range(first,stop):
            lo,hi=row_span(xy[a,0],xy[a,1],radii[a],lower,spacing,y,n);rows+=1
            if hi>lo:delta[y,lo]+=1;delta[y,hi]-=1
    count=0
    for y in range(n):
        active=0
        for x in range(n):active+=delta[y,x];count+=int(active>0)
    return count,rows

@njit(cache=True)
def tree_span(states,words,y,lo,hi,n,stack):
    """Online range OR; a full prefix stops traversal and siblings coalesce."""
    if hi<=lo:return 0,0
    leaves=words.shape[1];top=1;stack[0,0]=1;stack[0,1]=0;stack[0,2]=n
    visits=0;skipped=0
    while top:
        top-=1;node=stack[top,0];left=stack[top,1];right=stack[top,2]
        if node<0:
            node=-node
            if states[y,node*2]==2 and states[y,node*2+1]==2:states[y,node]=2
            continue
        visits+=1
        if states[y,node]==2:skipped+=1;continue
        if lo<=left and right<=hi:states[y,node]=2;continue
        if node>=leaves:
            idx=node-leaves
            words[y,idx]|=low_bits(min(hi,right)-left)^low_bits(max(lo,left)-left)
            states[y,node]=2 if words[y,idx]==low_bits(right-left) else 1
            continue
        states[y,node]=1;mid=(left+right)//2
        stack[top,0]=-node;stack[top,1]=left;stack[top,2]=right;top+=1
        if hi>mid:
            stack[top,0]=node*2+1;stack[top,1]=mid;stack[top,2]=right;top+=1
        if lo<mid:
            stack[top,0]=node*2;stack[top,1]=left;stack[top,2]=mid;top+=1
    return visits,skipped

@njit(cache=True)
def hierarchy_intervals(intervals):
    n=intervals.shape[1];leaves=(n+63)//64
    words=np.zeros((n,leaves),np.uint64);states=np.zeros((n,2*leaves),np.uint8)
    stack=np.empty((128,3),np.int64);visits=0;skips=0
    for a in range(len(intervals)):
        for y in range(n):
            v,s=tree_span(states,words,y,intervals[a,y,0],intervals[a,y,1],n,stack)
            visits+=v;skips+=s
    return states,words,visits,skips

@njit(cache=True)
def hierarchy_disks(xy,radii,lower,spacing,n):
    leaves=(n+63)//64;words=np.zeros((n,leaves),np.uint64);states=np.zeros((n,2*leaves),np.uint8)
    stack=np.empty((128,3),np.int64);rows=0;visits=0;skips=0
    for a in range(len(radii)):
        first,stop=row_bounds(xy[a,1],radii[a],lower,spacing,n)
        for y in range(first,stop):
            if states[y,1]==2:skips+=1;continue
            lo,hi=row_span(xy[a,0],xy[a,1],radii[a],lower,spacing,y,n);rows+=1
            v,s=tree_span(states,words,y,lo,hi,n,stack);visits+=v;skips+=s
    return states,words,rows,visits,skips

@njit(cache=True)
def compact_hierarchy(states,words,n):
    """Full prefix records (row,start,length); partial leaves (row,word,bits)."""
    leaves=words.shape[1];full=np.empty((n*leaves,3),np.int32)
    partial=np.empty((n*leaves,3),np.uint64);nf=0;np_=0;stack=np.empty((128,3),np.int64)
    for y in range(n):
        top=1;stack[0,0]=1;stack[0,1]=0;stack[0,2]=n
        while top:
            top-=1;node=stack[top,0];left=stack[top,1];right=stack[top,2]
            if states[y,node]==0:continue
            if states[y,node]==2:
                full[nf,0]=y;full[nf,1]=left;full[nf,2]=right-left;nf+=1
            elif node>=leaves:
                partial[np_,0]=y;partial[np_,1]=node-leaves;partial[np_,2]=words[y,node-leaves];np_+=1
            else:
                mid=(left+right)//2
                stack[top,0]=node*2+1;stack[top,1]=mid;stack[top,2]=right;top+=1
                stack[top,0]=node*2;stack[top,1]=left;stack[top,2]=mid;top+=1
    return full[:nf].copy(),partial[:np_].copy()

@njit(cache=True)
def prefix_count(full,partial):
    count=0
    for r in full:count+=int(r[2])
    for r in partial:count+=pop64(r[2])
    return count

@njit(cache=True)
def decode_prefixes(full,partial,n):
    words=np.zeros((n,(n+63)//64),np.uint64)
    for r in full:or_span(words,int(r[0]),int(r[1]),int(r[1]+r[2]))
    for r in partial:words[int(r[0]),int(r[1])]|=r[2]
    return words

@njit(cache=True)
def insert_word(states,words,y,k,bits,n):
    leaves=words.shape[1];node=leaves+k;parent=node
    while parent:
        if states[y,parent]==2:return
        parent//=2
    words[y,k]|=bits;states[y,node]=2 if words[y,k]==low_bits(min(n,64)) else 1
    node//=2
    while node:
        states[y,node]=2 if states[y,node*2]==2 and states[y,node*2+1]==2 else 1
        node//=2

@njit(cache=True)
def merge_prefix_arrays(full_a,partial_a,full_b,partial_b,n):
    leaves=(n+63)//64;states=np.zeros((n,2*leaves),np.uint8);words=np.zeros((n,leaves),np.uint64)
    stack=np.empty((128,3),np.int64)
    for full in (full_a,full_b):
        for r in full:tree_span(states,words,int(r[0]),int(r[1]),int(r[1]+r[2]),n,stack)
    for partial in (partial_a,partial_b):
        for r in partial:insert_word(states,words,int(r[0]),int(r[1]),r[2],n)
    return compact_hierarchy(states,words,n)

@njit(cache=True)
def weighted_prefix_sum(full,partial,prefix_weights):
    total=0.
    for r in full:total+=prefix_weights[r[0],r[1]+r[2]]-prefix_weights[r[0],r[1]]
    for r in partial:
        y=int(r[0]);base=int(r[1])*64;bits=r[2]
        for k in range(64):
            if (bits>>np.uint64(k))&np.uint64(1):total+=prefix_weights[y,base+k+1]-prefix_weights[y,base+k]
    return total


def unpack(words,n):
    return ((words[:,:,None]>>np.arange(64,dtype=np.uint64))&np.uint64(1)).astype(bool).reshape(n,-1)[:,:n]

@dataclass
class DyadicMask:
    n: int
    full: np.ndarray
    partial: np.ndarray

    @classmethod
    def from_intervals(cls,intervals):
        n=intervals.shape[1]
        if n<1 or n&(n-1):raise ValueError('Dyadic grid must be a power of two.')
        states,words,*_=hierarchy_intervals(intervals)
        return cls(n,*compact_hierarchy(states,words,n))

    @property
    def nbytes(self):return self.full.nbytes+self.partial.nbytes
    def count(self):return prefix_count(self.full,self.partial)
    def packed(self):return decode_prefixes(self.full,self.partial,self.n)
    def union(self,other):
        if self.n!=other.n:raise ValueError('Identical grid addresses required.')
        return DyadicMask(self.n,*merge_prefix_arrays(self.full,self.partial,other.full,other.partial,self.n))
    def weighted_sum(self,weights):
        if weights.shape!=(self.n,self.n):raise ValueError('Weight grid shape mismatch.')
        prefix=np.column_stack((np.zeros(self.n),np.cumsum(weights,axis=1)))
        return weighted_prefix_sum(self.full,self.partial,prefix)

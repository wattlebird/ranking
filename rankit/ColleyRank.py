import struct
import numpy as np
from solver import DefaultSolver
from BaseRank import BaseRank

class ColleyRank(BaseRank):
    """docstring for ColleyRank"""
    def __init__(self, arg):
        super(ColleyRank, self).__init__()
        self.arg = arg

    def rate(self, C, b):
        return DefaultSolver.solve(C,b)

    @classmethod
    def convert(cls, filename, filetype=1):
        with open(filename, "rb") as fr:
            if filetype==1:
                strrec = fr.read(8)
                if not strrec: return;
                ucnt, icnt = struct.unpack("ii", strrec)

                # allocate space for computing
                C = np.zeros((icnt,icnt), dtype=np.float32)
                w = np.zeros((icnt, 2), dtype=np.float32)

                curusr = ucnt;
                ilst=[];rlst=[];
                while True:
                    strrec = fr.read(10)
                    if not strrec: break;
                    uid, iid, rate = struct.unpack("iih", strrec)
                    if uid!=curusr:
                        # it is high time to summarize voting situation,
                        # for the last user
                        for i in xrange(len(ilst)):
                            for j in xrange(i, len(ilst)):
                                i1=ilst[i];i2=ilst[j]
                                C[i1, i2]-=1
                                C[i2, i1]-=1
                                if rlst[i]>rlst[j]:
                                    w[i1,0]+=1
                                    w[i2,1]+=1
                                elif rlst[i]<rlst[j]:
                                    w[i1,1]+=1
                                    w[i2,0]+=1
                                else:
                                    w[i1,1]+=0.5
                                    w[i2,1]+=0.5
                                    w[i1,0]+=0.5
                                    w[i2,0]+=0.5
                        #end double for
                        curusr=uid
                        ilst=[];rlst=[]
                    #end if
                    ilst.append(iid)
                    rlst.append(rate)
                # end while
                # The last user remains uncounted
                for i in xrange(len(ilst)):
                    for j in xrange(i, len(ilst)):
                        i1=ilst[i];i2=ilst[j]
                        C[i1, i2]-=1
                        C[i2, i1]-=1
                        if rlst[i]>rlst[j]:
                            w[i1,0]+=1
                            w[i2,1]+=1
                        elif rlst[i]<rlst[j]:
                            w[i1,1]+=1
                            w[i2,0]+=1
                        else:
                            w[i1,1]+=0.5
                            w[i2,1]+=0.5
                            w[i1,0]+=0.5
                            w[i2,0]+=0.5

                b = np.ravel(1+0.5*(w[:,0]-w[:,1]))
                for i in xrange(icnt):
                    C[i,i]=2+w[i,0]+w[i,1]
                return (C,b)
            # end if filetype
        # end with



def linvill_stability_factor(Yr,Yf,Gi,Go):
    YRYFcom = Yr*Yf
    YRYF = YRYFcom.real
    C = abs(Yr*Yf)/((2*Gi*Go)-YRYF)
    return C
def admitances_conductances(Yi,Yo,Yf,Yr):
    Gi = Yi.real
    Bi = Yi.imag
    Go = Yo.real
    Bo= Yo.imag
    Gf = Yf.real
    Bf = -Yf.imag
    Gr = Yr.real
    Br = Yr.imag
    return Gi,Bi,Go,Bo,Gf,Bf,Gr,Br


freq = 255e6
S11 = None
S12 = None
S21 = None
S22 = None
Yi = 2.25+7.2j
Yo = 0.4+1.9j
Yf = 40-20j
Yr = 0.05-0.7j
Gi = 8
Bi = 5.7
Go = 0.4
Bo= 1.5
Gf = 52
Bf = -20
Gr = 0.01
Br = -0.1
Im_unit = 1j
Zs = 50
Zl = 50
Ys = 1/Zs
Yl = 1/Zl
Gs = Ys.real
Gl = Yl.real
Bs = Ys.imag
Bl = Yl.imag

c = linvill_stability_factor(Yr,Yf,Gi,Go)
print(c)
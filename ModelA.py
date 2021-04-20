import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import integrate
import sympy

config={"font.family":"serif",
        "font.size":12,
        "mathtext.fontset":"stix",
        "font.serif":["SimSun"]}
rcParams.update(config)

# 参数解释
# m=2950e-3       #车体质量[kg]
# m1=2000e-3      #车头部质量[kg]
# m2=m-m1         #车轴部重量[kg]
# mw=100e-3       #车两轮质量[kg]
# L1=200e-3       #车头部高度[m]
# L2=190e-3       #车轴部高度[m]
# R=50e-3         #车轮半径[m]
# g=9.8           #重力加速度[m/s**2]

# I=m1*pow((L1/2+L2),2)+m2*L2*L2/12   #车身部分惯性[kg*m**2]
# Iw=2*(1/2)*mw*pow(R,2)              #车轮部分惯性[kg*m^2]
# Br=0.01                             #滚动阻尼比[N*m/(rad/s)]
# Bm=0.01                             #滑动阻尼比[N*m/(rad/s)]
# L=L2/2+(L1+L2)*m1/(2*m)             #重心位置[m]
# g=9.8                               #重力加速度[m/s**2]

# 数值解存储列表
lit_theta=[]
lit_phi=[]

def func(inv):
# 因变量设置
    t,g,m,mw,I,Iw,L,R,Br,Bm=sympy.symbols("t,g,m_v,m_w,I_v,I_w,L_v,R_v,beta_r,beta_m")

# 自变量设置
    theta,phi=sympy.symbols("theta_value,phi_value",cls=sympy.Function)

# 微分方程
    ode1=sympy.Eq((Iw+(mw+m)*R**2)*phi(t).diff(t,t)\
        +m*R*L*theta(t).diff(t,t)+(Br+Bm)*phi(t).diff(t)\
        -Bm*theta(t).diff(t),1.14)
# print(ode1)

    ode2=sympy.Eq(m*R*L*phi(t).diff(t,t)\
        +(I+m*L**2)*theta(t).diff(t,t)\
        -Bm*phi(t).diff(t)\
        +Bm*theta(t).diff(t)\
         -m*g*L*theta(t),-1.14)
# print(ode2)

# 微分方程降阶
    y1,y2,y3,y4=sympy.symbols("y_1,y_2,y_3,y_4",cls=sympy.Function)

    varchange={theta(t).diff(t,t):y2(t).diff(t),
            theta(t):y1(t),
            phi(t).diff(t,t):y4(t).diff(t), 
            phi(t):y3(t)}

    ode1_vc=ode1.subs(varchange)
    ode2_vc=ode2.subs(varchange)
    ode3=y1(t).diff(t)-y2(t)
    ode4=y3(t).diff(t)-y4(t)

    y=sympy.Matrix([y1(t),y2(t),y3(t),y4(t)])

    vcsol=sympy.solve((ode1_vc,ode2_vc,ode3,ode4),y.diff(t),dict=True)

    f=y.diff(t).subs(vcsol[0])

# print(sympy.Eq(y.diff(t),f))          #ODE函数f(t,y(t))的SymPy表达式

# 已知参数初始化
    params={m:2950e-3,
            mw:100e-3,
            I:0.171,
            Iw:0.25e-3,
            L:0.2272,
            R:50e-3,
            Br:0.01,
            Bm:0.01,
            g:9.8}

    _f_np= sympy.lambdify((t,y),f.subs(params),"numpy")
    f_np= lambda _t,_y,*args:_f_np(_t,_y)

    jac= sympy.Matrix([[fj.diff(yi) for yi in y] for fj in f])

    _jac_np=sympy.lambdify((t,y),jac.subs(params),"numpy")
    jac_np=lambda _t,_y,*args:_jac_np(_t,_y)

    y0=[inv,0,1.57,0]
    tt=np.linspace(0,10,10000)
    r=integrate.ode(f_np,jac_np).set_initial_value(y0,tt[0])
    dt=tt[1]-tt[0]
    yy=np.zeros((len(tt),len(y0)))

    idx=0
    while r.successful() and r.t<tt[-1]:
        yy[idx,:]=r.y
        r.integrate(r.t+dt)
        idx+=1

    lit_theta.append(yy[:,0])
    lit_phi.append(yy[:,2])

    return 0
    
def draw():
    tt=np.linspace(0,10,10000)
    lit_label=[5,10,15,20]
    lit_color=['k','r','y','g']
    plt.figure(figsize=(10,5))
    ax1=plt.subplot2grid((1,2),(0,0))
    ax2=plt.subplot2grid((1,2),(0,1))

    for j in range(4):
        ax1.plot(tt,lit_theta[j],lit_color[j],label="$\\theta$={}".format(lit_label[j]))
        ax2.plot(tt,lit_theta[j],lit_color[j],label="$\\theta$={}".format(lit_label[j]))
    ax1.set_xlabel("时间 t/s")
    ax2.set_xlabel("时间 t/s")
    ax1.set_ylabel("车身偏移角度 $\\theta$")
    ax2.set(xlim=[9,11])
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.savefig("theta.jpg")

    plt.figure(figsize=(10,5))
    ax3=plt.subplot2grid((1,2),(0,0))
    ax4=plt.subplot2grid((1,2),(0,1))

    for j in range(4):
        ax3.plot(tt,lit_phi[j],lit_color[j],label="$\\theta$={}".format(lit_label[j]))
        ax4.plot(tt,lit_phi[j],lit_color[j],label="$\\theta$={}".format(lit_label[j]))
    ax3.set_xlabel("时间 t/s")
    ax4.set_xlabel("时间 t/s")
    ax3.set_ylabel("车轮旋转角度 $\\phi$")
    ax4.set(xlim=[9,11])
    ax3.grid()
    ax4.grid()
    ax3.legend()
    ax4.legend()
    plt.savefig("phi.jpg")

    plt.show()

    return 0

def save():
    file_1=open("data_1.txt",'w',encoding="utf-8")
    file_2=open("data_2.txt",'w',encoding="utf-8")
    for i in range(len(lit_theta)):
        file_1.write(str(lit_theta[i]))
        file_1.write('\n')
    file_1.close()
    for j in range(len(lit_phi)):
        file_2.write(str(lit_phi[j]))
        file_2.write('\n')
    file_2.close()

    return 0

def main():
    lit_inv=[0.0873,0.1746,0.2616,0.3489]
    for i in range(len(lit_inv)):
        func(lit_inv[i])
    save()
    draw()

    return 0

if __name__=="__main__":
    main()

# 四阶龙格库塔法算法   
# def runge_kutta(y,x,dx,f):
#     k1=dx*f(y,x)
#     k2=dx*f(y+0.5*k1,x+0.5*dx)
#     k3=dx*f(y+0.5*k2,x+0.5*dx)
#     k4=dx*f(y+k3,x+dx)
#     return y+(k1+2*k2+2*k3+k4)/6

#include <Eigen/Dense>
#include <liegroup.hpp>

#include <quadrotor.hpp>
#include <iostream>

#include <ddp.hpp>
#include <simulator.hpp>

#include <string>

int main(int argc, char** argv)
{
	quadrotor::State xref;
	xref.R=SO3::exp((Vec3()<<0,0,0).finished());
	xref.xq=Vec3::Zero();
	xref.omega=Vec3::Zero();
	xref.vq=Vec3::Zero();


	std::vector<quadrotor::State> xrefs(4000,xref);
	std::vector<quadrotor::U> us(4000, 1*(Vec4()<<1,0,1,0).finished());

	std::srand((unsigned int) time(0));
	// Set up quadrotor parameters
	double m=0.6; // mass of the quadrotor
	double Ix=8*5e-3; // moment of inertia in X- Y- and Z-axis
	double Iy=7.5*5e-3;
	double Iz=13.5*5e-3;
	double d=0.2; // displacement of rotors
	double kt=0.6;
	double km=0.15;

	quadrotor::System sys(Ix,Iy,Iz,m,d,km,kt);

	// Set up cost function
	Mat12 Mf=5*Mat12::Identity()*10;
	Mf.block(0,0,3,3)*=2;
	Mf.block(3,3,3,3)*=6;
	Mf.block(6,6,6,6)=Mat6::Identity()*2;
	Mat12 M=Mf/2;
//	M.block(6,6,6,6)*=4;
	Mat4 R=Mat4::Identity()*2;
	DDP<quadrotor>::Params params(M,R,Mf);

	// Set up initial state
	quadrotor::State x0=xrefs[0];

	x0.xq-=(Vec3::Random()).normalized()*10;
	x0.R*=SO3::exp(Vec3::Random().normalized()*3);
	x0.omega=Vec3::Random().normalized()*1;
	x0.vq-=Vec3::Random().normalized()*1;
	// Set up simulator
	double dt=0.01;

	size_t num=200;
	Sim<quadrotor> sim(sys,dt);
	sim.init(x0,num);

	Vec4 umin=-Vec4::Ones()*3;
	Vec4 umax=Vec4::Ones()*3;

	DDP<quadrotor> ddp(sys,dt);

	int sn=1;
	int itr_max=200;
  for(int i=0;i<1000;i+=sn)
  {
    Vec12 error=quadrotor::State::diff(sim.get_state(),xrefs[0]);
    std::cout<<dt*i<<": "<<error.head(3).norm()<<" "<<error.head(6).tail(3).norm()<<std::endl;

    if(error.block(3,0,3,1).norm()<0.05 && error.head(3).norm()<0.4)
      break;

    double err=error.head(6).norm();
    if(err<2.5)
      itr_max=2000;
    else
      itr_max=1000;

    if(i==0)
      itr_max=10000;

    if(err<2)
    {
      M=2.5*Mat12::Identity()*10;
      M.block(0,0,3,3)*=2;
      M.block(3,3,3,3)*=6;
      M.block(6,6,6,6)=Mat6::Identity()*std::max(0.01, 1.25*err*err);

      Mf=5*Mat12::Identity()*100;
      Mf.block(0,0,3,3)*=2;
      Mf.block(3,3,3,3)*=6;
      Mf.block(6,6,6,6)=Mat6::Identity()*0.1;

      R=Mat4::Identity()*std::max(0.001, 0.5*err*err);
    }
    else
    {
      Mf=5*Mat12::Identity()*10;
      Mf.block(0,0,3,3)*=4;
      Mf.block(3,3,3,3)*=6;
      Mf.block(6,6,6,6)=Mat6::Identity()*5;
      M=Mf;
      R=Mat4::Identity()*1;
    }

    if(i%150==0 || (i%50==0 && us[0].norm()+us[10].norm()<0.5))
    {
      Vec4 u0=(Vec4()<<0.62,0.52,0.42,0.32).finished();
      for(int i=0;i<20;i++)
        us[i]=u0;
      ddp.init(sim.get_state(), us, xrefs, params, umin, umax, 150);
      ddp.iterate(40000,us);
    }
    else
    {
      timespec T_start, T_end;
      clock_gettime(CLOCK_MONOTONIC,&T_start);
      ddp.init(sim.get_state(), us, xrefs, params, umin, umax, 150);
      ddp.iterate(itr_max,us);
      clock_gettime(CLOCK_MONOTONIC,&T_end);
      std::cout<<"time consumed is "<<(T_end.tv_sec-T_start.tv_sec)+(T_end.tv_nsec-T_start.tv_nsec)/1000000000.0<<"s"<<std::endl;
    }

    for(int j=0;j<sn;j++)
    {
      sim.update(us.front());
      xrefs.erase(xrefs.begin());
      us.erase(us.begin());
    }
  }

  std::string path="/media/fantaosha/Documents/Northwestern/2016/Spring/Machine-Learning/Project/Data/quadrotor_ddp_1";
  if (argc>=2)
  {
    path.append("_");
    path.append(argv[1]);
  }
  path.append(".mat");
  sim.save(path);
}

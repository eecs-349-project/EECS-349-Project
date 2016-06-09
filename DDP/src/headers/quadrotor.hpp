#ifndef QUADROTOR
#define QUADROTOR

#include<iostream>
#include<cmath>
#include<string>
#include<functional>

#include<Eigen/Dense>
#include<mat.h>
#include<liegroup.hpp>

#include<vector>
#include<list>

struct quadrotor
{
	public:
		struct System;
		struct State;
		struct DState;

	public:
		struct System
		{
			const Mat3 I; // moment of inertia
			const double m; // mass

			const double d; // displacement of the motor
			const double km; // coefficient to balance the torque of the motor
			const double kt; // coefficient to generate lift force

			const Mat3 I_inv; // inverse of moment inertia

			System(double Ix, double Iy, double Iz, double m_, double d_, double km_, double kt_);
		};

	//**************************************************************************************************************************
		// Definition of quadrotor state and dynamics
	// *************************************************************************************************************************
		struct State // quadrotor state
		{
      Mat3 R; // SO(3) to represent orientation
      Vec3 xq; // posiition
      Vec3 omega; // angular velocity
			Vec3 vq; // linear velocity

			State();
			State(Mat4 g0, Vec6 v0);
			State(Mat3 R0, Vec3 x0, Vec3 w0, Vec3 v0);

			State update(const DState & dstate, double h); // compute next state by Euler method
			State update(const DState & dstate, Vec12 const & w, double h); // compute next state by Euler method
			static void  save(const std::vector<State> &, std::string);
			static Vec12 diff(State const & state, State const & ref);
		};

		struct DState // time derivative of quadrotor state
		{
			Vec3 omega; // angular velocity
			Vec3 vq; // linear velocity
      Vec3 alpha; // angular accelebration
      Vec3 aq; // linear accelebratoin

			DState(const System & sys, const State & state, Vec4 u); // compute body velocity
			DState(DState k1, DState k2, DState k3, DState k4); // compute average body velocity for RK4
		};

	public:
		typedef State Ref; // references used to compute cost function

		static constexpr double g=0; // accelebration of gravity

		static constexpr size_t M=12; // dimension of the configuration space
		static constexpr size_t N=4;  // dimension of control inputs

		typedef Eigen::Matrix<double, M,1> V;
		typedef Eigen::Matrix<double, N,1> U;


		static Eigen::Matrix<double,M,1> f(System const & sys, State const &state)
		{
			Eigen::Matrix<double,M,1> f;
      Vec3 const & omega=state.omega;

			f.block(0,0,3,1)=state.omega;
      f.block(3,0,3,1)=state.vq;
			f.block(6,0,3,1)=-sys.I_inv*SO3::hat(omega)*sys.I*omega;
			f.block(9,0,3,1)=-quadrotor::g*SO3::e[2];

			return f;
		}

		static Eigen::Matrix<double,M,N> h(System const & sys, State const & state)
		{
      Eigen::Matrix<double,quadrotor::M,quadrotor::N> H=Eigen::Matrix<double,quadrotor::M,quadrotor::N>::Zero();

      H.block(6,0,3,3)=sys.I_inv;

      H.block(9,3,3,1)=state.R.col(2)/sys.m;

      const static Eigen::Matrix<double,4,4> M=(Eigen::Matrix<double,4,4>()<<0,1,0,-1,-1,0,1,0,1,-1,1,-1,1,1,1,1).finished();

      Eigen::Matrix<double,4,4> MF=M;

      MF.block(0,0,2,4)*=sys.kt*sys.d;
      MF.block(2,0,1,4)*=sys.km;
      MF.block(3,0,1,4)*=sys.kt;

      H=H*MF;

			return H;
		}

		static Eigen::Matrix<double, M, M> Dgf(System const & sys, State const & state, U const & u)
		{
      Eigen::Matrix<double, M, M> df=Eigen::Matrix<double, M, M>::Zero();

      df.block(0,6,6,6)=Mat6::Identity();
			df.block(6,6,3,3)=sys.I_inv*(SO3::hat(sys.I*state.omega)-SO3::hat(state.omega)*sys.I);

      double f=sys.kt*(u(0)+u(1)+u(2)+u(3));
			df.block(9,0,3,3)=-f*state.R*SO3::E[2]/sys.m;

			return df;
		}

		static Eigen::Matrix<double,M,N> Duf(System const & sys, State const & state, U const & u)
		{
			return h(sys,state);
		}

		static Eigen::Matrix<double, M, M> ad(System const &sys, State const &state, U const & u)
		{
			Eigen::Matrix<double, M, M> ad=Eigen::Matrix<double,M,M>::Zero();
			ad.block(0,0,3,3)=SO3::ad(state.omega);

			return ad;
		}

		static void linearize (System const & sys, double const & dt,  State const & state, U const & u, Eigen::Matrix<double,12,12> & A, Eigen::Matrix<double,12,4> & B)
		{
      Mat3 Ad=SO3::Ad(SO3::exp(-state.omega*dt));
      Mat3 dexp=SO3::dexp(state.omega*dt);

			Mat12 dgF=quadrotor::Dgf(sys,state,u);
      dgF.block(0,6,3,6)=dexp*dgF.block(0,6,3,6);

			A=Mat12::Identity()+dgF*dt;
      A.block(0,0,3,12)=Ad*A.block(0,0,3,12);

			Eigen::Matrix<double,M,N> duF=quadrotor::Duf(sys,state, u);
			B=duF*dt;
		}

		static double L(const Mat12 &M, const Mat4 &R, const Vec12 &dg, const Vec4 &du)
		{

			return (dg.transpose()*M*dg+du.transpose()*R*du)(0)*0.5;
		}

		static Vec12 Lx(const Mat12 &M, const Vec12 &dg)
		{
			Vec12 Lx=M*dg;
			Lx.head(3)=SO3::dexpinv(-dg.head(3)).transpose()*Lx.head(3);

			return Lx;
		}

		static Mat12 Lxx(const Mat12 &M, const Vec12 &dg)
		{
			Vec3 dg_R=dg.head(3);
      Eigen::Matrix<double,9,1> dg_x=dg.tail(9);

			Mat3 dexpinv=SO3::dexpinv(-dg_R);
			Mat3 dexpinvT=dexpinv.transpose();


			Mat12 Lxx=M;
			Lxx.block(0,0,3,3)=dexpinvT*M.block(0,0,3,3)*dexpinv;
			Lxx.block(0,3,3,9)=dexpinvT*M.block(0,3,3,9);
			Lxx.block(3,0,9,3)=M.block(0,3,3,9).transpose();

/*************************************************************************
			Eigen::Matrix<double,1,3> r1=dg.transpose()*M.block(0,0,12,3);

			Eigen::Matrix<double,3,9> ddexpinv=SO3::ddexpinv(-err_R);

			Mat3 DM1=-dexpinvT*(Mat3()<<r1*ddexpinv.block(0,0,3,3),
										r1*ddexpinv.block(0,3,3,3),
										r1*ddexpinv.block(0,6,3,3)).finished();

			Eigen::Matrix<double,1,3> r2=-0.5*r1*dexpinv;

			Mat3 DM2=(Mat3()<<r2*SO3::ad(SO3::e[0]),
                        r2*SO3::ad(SO3::e[1]),
                        r2*SO3::ad(SO3::e[2])).finished();

			Lxx.block(0,0,3,3)+=DM1+DM2;
*************************************************************************/

			return Lxx;
		}

		static Vec4 Lu(const Mat4 &R, const Vec4 &du)
		{
			return R*du;
		}

		static Mat4 Luu(const Mat4 &R, const Vec4 &du)
		{
			return R;
		}

    static int load_u(std::string path, std::vector<U> & us, size_t const & sN, size_t const & SN)
    {
      MATFile *file=matOpen(path.c_str(),"r");

      if(file==NULL)
        return 0;
      else
      {
        mxArray *mxU;

        mxU=matGetVariable(file,"U");

        size_t N=mxGetN(mxU);
        N= SN*sN<N ? SN*sN:N;

        char *pU=(char *)mxGetPr(mxU);

        for(int i=0;i<N;i+=sN)
        {
          Eigen::Matrix<double,4,1> u;
          memcpy(u.data(),pU,sizeof(double)*u.rows()*u.cols());
          us.push_back(u.cwiseProduct(u));
          pU+=sizeof(double)*u.rows()*u.cols()*sN;
        }

        return 1;
      }
    }

    static int load_trajectory(std::string path, std::vector<State> & xs, std::vector<U> & us, size_t const & sN, size_t const & SN)
    {
      MATFile *file=matOpen(path.c_str(),"r");

      if(file==NULL)
        return 0;
      else
      {
        mxArray *mxR;
        mxArray *mxw;
        mxArray *mxxq;
        mxArray *mxvq;
        mxArray *mxU;

        mxR=matGetVariable(file,"state_R");
        mxw=matGetVariable(file,"state_w");
        mxxq=matGetVariable(file,"state_xq");
        mxvq=matGetVariable(file,"state_vq");

        mxU=matGetVariable(file,"U");

        size_t N=mxGetN(mxU);
        N= SN*sN<N ? SN*sN:N;

        char *pR=(char *)mxGetPr(mxR);
        char *pw=(char *)mxGetPr(mxw);
        char *pxq=(char *)mxGetPr(mxxq);
        char *pvq=(char *)mxGetPr(mxvq);
        char *pU=(char *)mxGetPr(mxU);

        xs.clear();
        xs.reserve(N/sN+1);

        for(int i=0;i<N;i+=sN)
        {
          Eigen::Matrix<double,3,3> R;
          Eigen::Matrix<double,3,1> w,xq,vq;

          memcpy(R.data(),pR,sizeof(double)*R.rows()*R.cols());
          memcpy(w.data(),pw,sizeof(double)*w.rows()*w.cols());
          memcpy(xq.data(),pxq,sizeof(double)*xq.rows()*xq.cols());
          memcpy(vq.data(),pvq,sizeof(double)*vq.rows()*vq.cols());

          xs.emplace_back(R,xq,w,vq);

          pR+=sizeof(double)*R.rows()*R.cols()*sN;
          pw+=sizeof(double)*w.rows()*w.cols()*sN;
          pxq+=sizeof(double)*xq.rows()*xq.cols()*sN;
          pvq+=sizeof(double)*vq.rows()*vq.cols()*sN;
        }

        for(int i=0;i<N;i+=sN)
        {
          Eigen::Matrix<double,4,1> u;
          memcpy(u.data(),pU,sizeof(double)*u.rows()*u.cols());
          us.push_back(u.cwiseProduct(u));
          pU+=sizeof(double)*u.rows()*u.cols()*sN;
        }

        return 1;
      }
    }

    static std::vector<Eigen::Matrix<double,4,12> > lqr(System const &sys, double dt, std::vector<State> const xs, std::vector<U> const & us, Mat12 Q, Mat4 Ru, Mat12 Qf)
    {
      std::vector<Eigen::Matrix<double,4,12> > Ks;

      std::vector<State>::const_reverse_iterator rit_state=xs.crbegin();
      std::vector<U>::const_reverse_iterator rit_u=us.crbegin();

      rit_state++;

      Mat4 MF=(Mat4()<<             0,  sys.kt*sys.d,             0, -sys.kt*sys.d,
                        -sys.kt*sys.d,             0,  sys.kt*sys.d,             0,
                               sys.km,       -sys.km,        sys.km,       -sys.km,
                               sys.kt,        sys.kt,        sys.kt,        sys.kt).finished();

      Mat12 P=Qf;

      while(rit_state!=xs.crend())
      {
        Vec4 u=*rit_u;
        Vec4 u2=u.cwiseProduct(u);
        Mat3 R=rit_state->R;
        Vec3 omega=rit_state->omega;

        Mat3 Ad=SO3::exp(-omega*dt);

        double f=(u2(0)+u2(1)+u2(2)+u2(3))*sys.kt;

        Mat12 A=Mat12::Zero();
        A.block(0,6,6,6)=Mat6::Identity();
        A.block(6,6,3,3)=sys.I_inv*(SO3::hat(sys.I*omega)-SO3::hat(omega)*sys.I);
        A.block(9,0,3,3)=-f*R*SO3::E[2]/sys.m;
        A.block(0,6,3,6)=SO3::dexp(omega*dt)*A.block(0,6,3,6);

        A=Mat12::Identity()+A*dt;
        A.block(0,0,3,12)=SO3::exp(-omega*dt)*A.block(0,0,3,12);

        Eigen::Matrix<double,12,4> B=Eigen::Matrix<double,12,4>::Zero();
        B.block(6,0,3,3)=sys.I_inv;
        B.block(9,3,3,1)=R.col(2)/sys.m;
        B=B*MF*(2*u).asDiagonal();
        B=B*dt;

        Eigen::Matrix<double,4,12> K=(Ru+B.transpose()*P*B).inverse()*B.transpose()*P*A;
        P=K.transpose()*Ru*K+Q+(A-B*K).transpose()*P*(A-B*K);

        Ks.push_back(K);

        rit_state++;
        rit_u++;
      }

      std::reverse(Ks.begin(),Ks.end());
      return Ks;
      }
};

quadrotor::System::System(double Ix, double Iy, double Iz, double m_, double d_, double km_, double kt_): I((Mat3()<<Ix,0,0,0,Iy,0,0,0,Iz).finished()), m(m_), d(d_), km(km_), kt(kt_), I_inv((Mat3()<<1.0/Ix, 0, 0, 0, 1.0/Iy,0,0, 0, 1.0/Iz).finished())
{
}


quadrotor::State::State():R(Mat3::Identity()), xq(Vec3::Zero()), omega(Vec3::Zero()), vq(Vec3::Zero())
{
}

quadrotor::State::State(Mat4 g0, Vec6 v0):R(g0.block(0,0,3,3)), xq(g0.block(0,3,3,1)), omega(v0.head(3)), vq(g0.block(0,0,3,3)*v0.tail(3))
{
}

quadrotor::State::State(Mat3 R0, Vec3 x0, Vec3 w0, Vec3 v0): R(R0), xq(x0), omega(w0), vq(v0)
{
}

quadrotor::State quadrotor::State::update(const DState & dstate, double h)
{
	State state_next;

	state_next.R=R*SO3::exp(dstate.omega*h);
  state_next.xq=xq+dstate.vq*h;
  state_next.omega=omega+dstate.alpha*h;
	state_next.vq=vq+dstate.aq*h;

	return state_next;
}

quadrotor::State quadrotor::State::update(const DState & dstate, Vec12 const & w, double h)
{
	State state_next;

  double hsqrt = sqrt(h);

	state_next.R=R*SO3::exp(dstate.omega*h+hsqrt*w.head(3));
  state_next.xq=xq+dstate.vq*h+hsqrt*w.block(3,0,3,1);
  state_next.omega=omega+dstate.alpha*h+hsqrt*w.block(6,0,3,1);
	state_next.vq=vq+dstate.aq*h+hsqrt*w.tail(3);

	return state_next;
}

void quadrotor::State::save(const std::vector<State> & states, std::string path)
{
	MATFile *result;

	mxArray *R;
  mxArray *theta;
	mxArray *xq;
	mxArray *omega;
	mxArray *vq;

	char *p;

	mwSize *dims=new mwSize[3];

	result=matOpen(path.c_str(),"w");

	std::list<State>::const_iterator it_state;
  std::list<State> list_state(states.begin(),states.end());

	it_state=list_state.cbegin();
	dims[0]=3;
	dims[1]=3;
	dims[2]=states.size();

	R=mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);


	for(p=(char*)mxGetPr(R);it_state!=list_state.cend();it_state++)
	{
		memcpy((void*)p,it_state->R.data(),sizeof(double)*dims[0]*dims[1]);
		p+=sizeof(double)*dims[0]*dims[1];
	}

	matPutVariable(result,"R",R);
	mxDestroyArray(R);

	it_state=list_state.cbegin();
	dims[0]=3;
	dims[1]=1;
	dims[2]=states.size();

	theta=mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);


	for(p=(char*)mxGetPr(theta);it_state!=list_state.cend();it_state++)
	{
		memcpy((void*)p,SO3::log(it_state->R).data(),sizeof(double)*dims[0]*dims[1]);
		p+=sizeof(double)*dims[0]*dims[1];
	}

	matPutVariable(result,"theta",theta);
	mxDestroyArray(R);

	it_state=list_state.cbegin();
	dims[0]=3;
	dims[1]=1;
	dims[2]=states.size();

	xq=mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);


	for(p=(char*)mxGetPr(xq);it_state!=list_state.cend();it_state++)
	{
		memcpy((void*)p,it_state->xq.data(),sizeof(double)*dims[0]*dims[1]);
		p+=sizeof(double)*dims[0]*dims[1];
	}

	matPutVariable(result,"xq",xq);
	mxDestroyArray(xq);

	it_state=list_state.cbegin();
	dims[0]=3;
	dims[1]=1;
	dims[2]=states.size();

	vq=mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);


	for(p=(char*)mxGetPr(vq);it_state!=list_state.cend();it_state++)
	{
		memcpy((void*)p,it_state->vq.data(),sizeof(double)*dims[0]*dims[1]);
		p+=sizeof(double)*dims[0]*dims[1];
	}

	matPutVariable(result,"vq",vq);
	mxDestroyArray(vq);

	it_state=list_state.cbegin();
	dims[0]=3;
	dims[1]=1;
	dims[2]=states.size();

	omega=mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);


	for(p=(char*)mxGetPr(omega);it_state!=list_state.cend();it_state++)
	{
		memcpy(p,it_state->omega.data(),sizeof(double)*dims[0]*dims[1]);
		p+=sizeof(double)*dims[0]*dims[1];
	}

	matPutVariable(result,"omega",omega);
	mxDestroyArray(omega);
}

Vec12 quadrotor::State::diff(State const & x, State const & xref)
{
  Vec12 error=(Vec12()<<SO3::log(xref.R.transpose()*x.R),
              x.xq-xref.xq,
              x.omega-xref.omega,
              x.vq-xref.vq).finished();

	return error;
}

quadrotor::DState::DState(const System & sys, const State & state, Vec4 u)
{
	Eigen::Matrix<double,M,1> dstate=f(sys,state)+h(sys,state)*u;

  omega=dstate.head(3);
  vq=dstate.block(3,0,3,1);
  alpha=dstate.block(6,0,3,1);
  aq=dstate.tail(3);
}

quadrotor::DState::DState(DState k1, DState k2, DState k3, DState k4)
{
	omega=(k1.omega+2*k2.omega+2*k3.omega+k4.omega)/6;
	vq=(k1.vq+2*k2.vq+2*k3.vq+k4.vq)/6;
	alpha=(k1.alpha+2*k2.alpha+2*k3.alpha+k4.alpha)/6;
	aq=(k1.aq+2*k2.aq+2*k3.aq+k4.aq)/6;
}
#endif

#include "../core/scalar.h"
#ifndef SE3mat_H
#define SE3mat_H

#include<eigen3/Eigen/Geometry>
#include<eigen3/Eigen/Core>

namespace g2o {

class SE3mat
{
public:
    SE3mat(){
        R = Matrix3::Identity();
        t.setZero();
    }

    SE3mat(const Matrix3 &R_, const Vector3 &t_):R(R_),t(t_){}

    void Retract(const Vector3 dr, const Vector3 &dt);

    inline Vector3 operator* (const Vector3& v) const {
      return R*v + t;
    }

    inline SE3mat& operator*= (const SE3mat& T2){
      t+=R*T2.t;
      R*=T2.R;
      return *this;
    }

    inline SE3mat inverse() const{
      Matrix3 Rt = R.transpose();
      return SE3mat(Rt,-Rt*t);
    }

protected:
    Vector3 t;
    Matrix3 R;

public:
    static Matrix3 ExpSO3(const Vector3 r);
    static Vector3 LogSO3(const Matrix3 R);
};

}//namespace g2o

#endif // SE3mat_H

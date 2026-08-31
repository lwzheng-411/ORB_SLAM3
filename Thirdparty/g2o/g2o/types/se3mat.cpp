#include "../core/scalar.h"
#include "se3mat.h"

namespace g2o {


void SE3mat::Retract(const Vector3 dr, const Vector3 &dt)
{
    t += R*dt;
    R = R*ExpSO3(dr);
}

Matrix3 SE3mat::ExpSO3(const Vector3 r)
{
    Matrix3 W;
    W << 0, -r[2], r[1],
         r[2], 0, -r[0],
         -r[1], r[0], 0;

    const number_t theta = r.norm();

    if(theta<1e-6)
        return Matrix3::Identity() + W + 0.5l*W*W;
    else
        return Matrix3::Identity() + W*sin(theta)/theta + W*W*(1-cos(theta))/(theta*theta);
}

Vector3 SE3mat::LogSO3(const Matrix3 R)
{
    const number_t tr = R(0,0)+R(1,1)+R(2,2);
    const number_t theta = acos((tr-1.0l)*0.5l);
    Vector3 w;
    w << R(2,1), R(0,2), R(1,0);
    if(theta<1e-6)
        return w;
    else
        return theta*w/sin(theta);
}

} //namespace g2o

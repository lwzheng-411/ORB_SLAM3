// Build-selectable scalar type for the whole g2o pipeline.
//
// With G2O_SCALAR_FLOAT defined, every residual / Jacobian / Hessian /
// linear-solve / LM inner-loop value runs in fp32, while map state owned
// by the application may stay double (oplusImpl adds a float increment to
// a double estimate). Without the define, number_t = double and the
// library behaves exactly as the original code, so the stock libg2o.so
// build is unaffected.
//
// Wall-clock time (stuff/timeutil.*, stuff/os_specific.*) intentionally
// stays double: epoch-scale timestamps do not survive fp32.
#ifndef G2O_CORE_SCALAR_H
#define G2O_CORE_SCALAR_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {

#ifdef G2O_SCALAR_FLOAT
  typedef float number_t;
#else
  typedef double number_t;
#endif

  typedef Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
  typedef Eigen::Matrix<number_t, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<number_t, 2, 1> Vector2;
  typedef Eigen::Matrix<number_t, 3, 1> Vector3;
  typedef Eigen::Matrix<number_t, 4, 1> Vector4;
  typedef Eigen::Matrix<number_t, 2, 2> Matrix2;
  typedef Eigen::Matrix<number_t, 3, 3> Matrix3;
  typedef Eigen::Matrix<number_t, 4, 4> Matrix4;
  typedef Eigen::Transform<number_t, 3, Eigen::Isometry> Isometry3;

} // namespace g2o

#endif // G2O_CORE_SCALAR_H

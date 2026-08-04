#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#define MAT_AT(m, i, j) (m).start[(i)*(m).cols + (j)]
#define MAT_PRINT(m) mat_print(m, #m)

typedef struct {
   int rows;
   int cols;
   float *start;
} mat;

mat mat_alloc(int rows, int cols)
{
   mat m;
   m.rows = rows;
   m.cols = cols;
   m.start = malloc(sizeof(*m.start)*rows*cols);
   assert(m.start != NULL);
   return m;
}

void mat_free(mat m)
{
   free(m.start);
}

void mat_fill(mat dst, float num)
{
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) = num;
      }
   }
}

void mat_print(mat m, char* name) 
{
   printf("%s = [\n", name);
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         printf(" %f ", MAT_AT(m, i, j));
      }
      printf("\n");
   }
   printf("]\n");
}

void mat_add(mat dst, mat src)
{
   assert(dst.rows == src.rows);
   assert(dst.cols == src.cols);
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) += MAT_AT(src, i, j);
      }
   }
}

void mat_add_broadcast(mat dst, mat scalar)
{
   // make sure scalar is a single val just in mat form
   assert(scalar.rows == 1 && scalar.cols == 1);
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) += scalar.start[0];
      }
   }

}

void mat_diff(mat dst, mat src)
{
   assert(dst.rows == src.rows);
   assert(dst.cols == src.cols);
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) -= MAT_AT(src, i, j);
      }
   }
}

void mat_sq(mat dst)
{
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) = MAT_AT(dst, i, j) * MAT_AT(dst, i, j);
      }
   }
}

void mat_mult_elem(mat dst, mat a, mat b)
{
   assert(dst.rows == a.rows && a.rows == b.rows);
   assert(dst.cols == a.cols && a.cols == b.cols);
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
      }
   }
}

// a dot b, store in dst
void mat_mult(mat dst, mat a, mat b)
{
   assert(a.cols == b.rows);
   assert(dst.rows == a.rows);
   assert(dst.cols == b.cols);

   // [x n] [n x]
   // this is the COMMON dimension for a & b
   // **
   // col for a
   // row for b
   size_t n = a.cols;

   for (int i = 0 ; i < dst.rows ; ++i) {   
      for (int j = 0 ; j < dst.cols ; ++j) {

         MAT_AT(dst, i, j) = 0;

         for (size_t k = 0 ; k < n ; ++k) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
         }
      }
      
   }
}

float mat_sum(mat m)
{
   float res = 0;
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         res += MAT_AT(m, i, j);
      }
   }
   return res;
}

void sigmoid(mat m) {
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         MAT_AT(m, i, j) = 1 / (1 + exp(-MAT_AT(m, i, j)));
      }
   }
}

void mat_mult_scalar(mat m, float x)
{
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         MAT_AT(m, i, j) = MAT_AT(m, i, j) * x;
      }
   }
}

float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

void rand_init(mat m) 
{
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         MAT_AT(m, i, j) = random_float();
      }
   }
}

mat transpose(mat m)
{
   int rows = m.rows;
   int cols = m.cols;
   mat mt = mat_alloc(cols, rows);

   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         MAT_AT(mt, j, i) = MAT_AT(m, i, j);
      }
   }

   return mt;
}

void mat_copy(mat dst, mat src)
{
   assert(dst.rows == src.rows);
   assert(dst.cols == src.cols);
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) = MAT_AT(src, i, j);
      }
   }
}

// derivative of sigmoid, given m already holds sigmoid(x): sig'(x) = sig(x) * (1 - sig(x))
void sig_grad(mat m)
{
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         MAT_AT(m, i, j) = MAT_AT(m, i, j) * (1 - MAT_AT(m, i, j));
      }
   }
}

int main() {
   srand(time(NULL));
   float in[] = {
      0, 0,
      0, 1,
      1, 0,
      1, 1
      };
   float out[] = {
      0,
      0, 
      0, 
      1
      };

   mat x = { .rows = 4, .cols = 2, .start = in };
   mat y = { .rows = 4, .cols = 1, .start = out };
   mat w = mat_alloc(2, 1);
   rand_init(w);
   mat b = mat_alloc(1, 1);
   rand_init(b);

   MAT_PRINT(x);
   MAT_PRINT(y);
   MAT_PRINT(w);
   MAT_PRINT(b);

   float learning_rate = 1e-3;
   int epochs = 100000;

   mat a1 = mat_alloc(4, 1);
   mat cost = mat_alloc(4, 1);
   mat sig_deriv = mat_alloc(4, 1);
   mat dcost = mat_alloc(4, 1);
   mat xt = transpose(x); // x doesn't change, so its transpose is computed once
   mat dcost_w = mat_alloc(2, 1);
   mat dcost_b = mat_alloc(1, 1);

   for (int epoch = 0 ; epoch < epochs ; ++epoch) {
      // forward
      mat_mult(a1, x, w); // a1 = xw
      mat_add_broadcast(a1, b); // a1 = xw + b
      sigmoid(a1); // a1 = sig(a1)

      // cost
      mat_copy(cost, y);
      mat_diff(cost, a1); // y - a1
      mat_sq(cost); // (y - a1)**2
      mat_mult_scalar(cost, 0.25); // (1 / n) (y - a1)**2

      // d cost WITH RESPECT TO W : 1/n * 2 * x.T * ((a1 - y) * sig'(z))
      // where a1 = sig(z), z = xw + b, sig'(z) = a1 * (1 - a1)
      mat_copy(sig_deriv, a1);
      sig_grad(sig_deriv); // sig_deriv = a1 * (1 - a1)

      mat_copy(dcost, a1); // dcost = a1
      mat_diff(dcost, y); // dcost = a1 - y
      mat_mult_elem(dcost, dcost, sig_deriv); // dcost = (a1 - y) * a1 * (1 - a1)

      mat_mult(dcost_w, xt, dcost); // x.T * dcost -> [2 1]
      mat_mult_scalar(dcost_w, 0.5); // * 1/n * 2

      // d cost WITH RESPECT TO B : 1/n * 2 * (a1 - y) * sig'(z)
      // dz/db = 1 (b is added elementwise to z), so no x.T needed here

      MAT_AT(dcost_b, 0, 0) = mat_sum(dcost) * 0.5;

      // update weights and bias
      mat_mult_scalar(dcost_w, learning_rate);
      mat_diff(w, dcost_w);

      mat_mult_scalar(dcost_b, learning_rate);
      mat_diff(b, dcost_b);

      if (epoch % 10000 == 0) {
         float total_cost = 0;
         for (int i = 0 ; i < cost.rows ; ++i) total_cost += MAT_AT(cost, i, 0);
         printf("epoch %d: cost = %f\n", epoch, total_cost);
      }
   }

   printf("\nfinal:\n");
   MAT_PRINT(w);
   MAT_PRINT(b);
   MAT_PRINT(a1); // predictions, compare against y

   mat_free(a1);
   mat_free(cost);
   mat_free(sig_deriv);
   mat_free(dcost);
   mat_free(xt);
   mat_free(dcost_w);
   mat_free(dcost_b);
   mat_free(w);
   mat_free(b);

   return 0;
}
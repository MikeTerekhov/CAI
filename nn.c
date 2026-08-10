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
   // rows are batch examples and columns are features/units
   // A bias always has one value per output unit, which needs to be added identically to every example in the batch
   assert(scalar.rows == 1 && scalar.cols == dst.cols);
   for (int i = 0 ; i < dst.rows ; ++i) {
      for (int j = 0 ; j < dst.cols ; ++j) {
         MAT_AT(dst, i, j) += MAT_AT(scalar, 0, j);
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

// dst is [1, src.cols]
//dst = [..............]
// sums across all the ROWS of SRC and puts it into dst row vec
void mat_col_sum(mat dst, mat src)
{
   assert(dst.rows == 1);
   assert(dst.cols == src.cols);
   for (int j = 0 ; j < src.cols ; j++){
      float sum = 0;
      for (int i = 0 ; i < src.rows ; i++) {
          sum += MAT_AT(src, i, j);
      }
      MAT_AT(dst, 0, j) = sum;
   }
}

void sigmoid(mat m) {
   for (int i = 0 ; i < m.rows ; ++i) {
      for (int j = 0 ; j < m.cols ; ++j) {
         MAT_AT(m, i, j) = 1 / (1 + exp(-MAT_AT(m, i, j)));
      }
   }
}

// single point prediction helper
float predict(float x1, float x2, mat w, mat b)
{
   // z = x1 * w + x2 * w + b
   float z = x1 * MAT_AT(w, 0, 0) + x2 * MAT_AT(w, 1, 0) + MAT_AT(b, 0, 0);
   return 1 / (1 + exp(-z));
}

void print_decision_boundary(mat w, mat b, int resolution)
{
   for (int i = 0 ; i < resolution ; i++) {
      float x2 = 1.0f - (float)i / (resolution - 1);
      for (int j = 0 ; j < resolution ; j ++) {
         float x1 = (float)j / (resolution - 1); 
         float pred = predict(x1, x2, w, b);
         putchar(pred > 0.5f ? '#' : '.');
      }
      putchar('\n');
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

void mat_transpose_into(mat dst, mat src)
{
   assert(dst.rows == src.cols);
   assert(dst.cols == src.rows);
   for (int i = 0 ; i < src.rows ; ++i) {
      for (int j = 0 ; j < src.cols ; ++j) {
         MAT_AT(dst, j, i) = MAT_AT(src, i, j);
      }
   }
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
      1, 
      1, 
      0
      };

   // input and output
   mat x = { .rows = 4, .cols = 2, .start = in };
   mat y = { .rows = 4, .cols = 1, .start = out };

   // weights
   int H = 2;
   mat w1 = mat_alloc(2, H); // input -> hidden
   rand_init(w1);
   mat b1 = mat_alloc(1, H); // one bias per unit
   rand_init(b1);

   mat w2 = mat_alloc(H, 1); // hiden -> output
   rand_init(w2);
   mat b2 = mat_alloc(1, 1); // single output bias
   rand_init(b2);

   MAT_PRINT(x);
   MAT_PRINT(y);

   MAT_PRINT(w1);
   MAT_PRINT(b1);
   MAT_PRINT(w2);
   MAT_PRINT(b2);

   float learning_rate = 1e-3;
   int epochs = 100000;

   //---------------------------------------------------
   mat hidden = mat_alloc(4, H);       // hidden layer activations
   mat pred = mat_alloc(4, 1);          // final output

   mat cost = mat_alloc(4, 1);
   mat sig_deriv_out = mat_alloc(4, 1);      // out * (1 - out)
   mat sig_deriv_hidden = mat_alloc(4, H);   // hidden * (1 - hidden)

   mat dcost = mat_alloc(4, 1);        // error signal at the output layer
   mat dhidden = mat_alloc(4, H);      // error signal at the hidden layer

   mat xt = transpose(x);              // constant, computed once: [2,4]
   mat ht = mat_alloc(H, 4);           // hidden.T, recomputed every epoch (hidden changes)
   mat w2t = mat_alloc(1, H);          // w2.T, recomputed every epoch (w2 changes)

   mat dcost_w1 = mat_alloc(2, H);
   mat dcost_b1 = mat_alloc(1, H);
   mat dcost_w2 = mat_alloc(H, 1);
   mat dcost_b2 = mat_alloc(1, 1);

   for (int epoch = 0 ; epoch < epochs ; ++epoch) {
      // forward
      mat_mult(hidden, x, w1);      // hidden = x @ w1
      mat_add_broadcast(hidden, b1);
      sigmoid(hidden);

      mat_mult(pred, hidden, w2);    // out = hidden @ w2
      mat_add_broadcast(pred, b2);
      sigmoid(pred);

      //
      // cost
      mat_copy(cost, y);
      mat_diff(cost, pred); // y - pred
      mat_sq(cost);
      mat_mult_scalar(cost, 0.25);

      // d cost w.r.t. output layer (w2, b2)
      mat_copy(sig_deriv_out, pred);
      sig_grad(sig_deriv_out); // pred * (1 - pred)

      mat_copy(dcost, pred);
      mat_diff(dcost, y); // dcost = pred - y
      mat_mult_elem(dcost, dcost, sig_deriv_out); // dcost = (pred - y) * pred * (1 - pred)

      mat_transpose_into(ht, hidden); // ht = hidden.T  [H,4]
      mat_mult(dcost_w2, ht, dcost);  // hidden.T * dcost -> [H,1]
      mat_mult_scalar(dcost_w2, 0.5);

      MAT_AT(dcost_b2, 0, 0) = mat_sum(dcost) * 0.5;

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

   printf("\ndecision boundary (x1 -->, x2 ^):\n");
   print_decision_boundary(w, b, 21);


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
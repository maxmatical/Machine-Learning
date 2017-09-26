# derivative function (relu)
relu_deriv = function(x){
  x_deriv = matrix(0, nrow = dim(x)[1], ncol = dim(x)[2])  
  for( j in 1:dim(x)[1]){
    for( k in 1:dim(x)[2]){
      if(x[j,k] <=0){x_deriv[j,k] = 0}
      else {x_deriv[j,k] = 1}
    }
  }
  return (x_deriv)
}
#derivative function selu

selu_deriv = function(x){
  x_deriv = matrix(0, nrow = dim(x)[1], ncol = dim(x)[2])  
  lambda = 1.0507
  alpha = 1.6733
  for( j in 1:dim(x)[1]){
    for( k in 1:dim(x)[2]){
      if(x[j,k] <=0){x_deriv[j,k] = lambda}
      else {x_deriv[j,k] = lambda*alpha*exp(x[j,k])}
    }
  }
  return (x_deriv)
}
#testing
test = matrix(rnorm(25, mean = 0, sd = 5), nrow = 5, ncol=5)
test
relu_deriv(test)
selu_deriv(test)
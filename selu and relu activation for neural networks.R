# defining constants
lambda = 1.0507
alpha = 1.6733

#relu function
relu = function(x){
  x_relu = matrix(0, nrow = dim(x)[1], ncol = dim(x)[2])  
  for( j in 1:dim(x)[1]){
    for( k in 1:dim(x)[2]){
      if(x[j,k] <=0){x_relu[j,k] = 0}
      else {x_relu[j,k] = x[j,k]}
    }
  }
  return (x_relu)
}

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
# selu
selu = function(x, lambda, alpha){
  x_selu = matrix(0, nrow = dim(x)[1], ncol = dim(x)[2])  
  for( j in 1:dim(x)[1]){
    for( k in 1:dim(x)[2]){
      if(x[j,k] >0){x_selu[j,k] = lambda*x[j,k]} 
      else {x_selu[j,k] = lambda*alpha*(exp(x[j,k])-1)}
    }
  }
  return (x_selu)
}

#derivative function selu
selu_deriv = function(x, lambda, alpha){
  x_deriv = matrix(0, nrow = dim(x)[1], ncol = dim(x)[2])  
  for( j in 1:dim(x)[1]){
    for( k in 1:dim(x)[2]){
      if(x[j,k] >0){x_deriv[j,k] = lambda} 
      else {x_deriv[j,k] = lambda*alpha*exp(x[j,k])}
    }
  }
  return (x_deriv)
}
#testing
test = matrix(rnorm(25, mean = 0, sd = 5), nrow = 5, ncol=5)
test
# relu/selu
relu(test)
selu(test,lambda,alpha)
#derivative matrix
relu_deriv(test)
selu_deriv(test, lambda, alpha)


# defining training x, y
x = matrix(c(0,0,1,0,1,1,1,0,1,1,1,1), ncol = 3, nrow = 4, byrow=T)
y = matrix(c(0, 1, 1, 0), ncol = 1, nrow = 4, byrow=T)

# defining # of hidden units in each layer
n_units_l1 = 5
n_units_l2= 5
dim_input = 3

# different initializers
# set sd
sd_initial = 0.001
# glorot: 
se_glorot0 = sqrt(2/(dim_input+n_units_l1))
se_glorot1 = sqrt(2/(n_units_l2+n_units_l1))
se_glorot2 = sqrt(2/(n_units_l2+1))
# he
se_he0 = sqrt(2/dim_input)
se_he1 = sqrt(2/n_units_l1)
se_he2 = sqrt(2/n_units_l2)
# lecun
se_le0 = sqrt(1/dim_input)
se_le1 = sqrt(1/n_units_l1)
se_le2 = sqrt(1/n_units_l2)
#initialize weights
w0 = matrix(rnorm(n_units_l1*dim_input, mean=0, sd = se_glorot0), 
            ncol= n_units_l1, nrow = dim_input) #dim_input X n_units

w1 = matrix(rnorm(n_units_l1*n_units_l2, mean=0, sd = se_glorot1), 
            ncol= n_units_l1, nrow = n_units_l2) #2nd weights

w2 = matrix(rnorm(n_units_l2*1, mean=0, sd = se_glorot2), 
            ncol= n_units, nrow = 1) #1 output per obs #n_units x 1



#initialize learning rate
learn_rate = 0.001
# initialize for adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08
m = 0
v = 0

# training NN on  epochs
epochs = 60000
for (i in 1:epochs){
  l1 = 1/(1+exp(-x%*%w0)) # 1st hidden layer  
  # adding dropout
  A1 = sample(1:dim(l1)[2], dim(l1)[2]/2, replace = F)
  l1[,A1] = 0
  #connecting to 2nd layer
  l2 = 1/(1+exp(-l1%*%t(w1))) # 2nd hidden layer
  # droput for 2nd hidden layer
  A2 = sample(1:dim(l2)[2], dim(l2)[2]/2, replace = F)
  l2[,A2] = 0
  # connect to output layer
  l3 = 1/(1+exp(-l2%*%t(w2))) #output layer
  #error 
  l3_delta = (y-l3)*(l3*(1-l3)) #find delta and multiply by chain rule (weights*(1-weights) is gradient for sigmoid)
  l2_delta = l3_delta%*%w2*(l2*(1-l2)) 
  l1_delta = l2_delta%*%w1*(l1*(1-l1))
  w2 = w2 + t(learn_rate*t(l2)%*%l3_delta)
  w1 = w1 + t(learn_rate*t(l1)%*%l2_delta)
  w0 = w0 + learn_rate*t(x)%*%l1_delta
}
  
# test prediction
x_test = matrix(c(1,0,0),ncol = 3, nrow = 1) # define x_test
l1_pred = 1/(1+exp(-x_test%*%w0)) #go through layer 1 (hidden layer)
l2_pred = 1/(1+exp(-l1_pred%*%t(w1)))
output_proba = 1/(1+exp(-l2_pred%*%t(w2))) # go through output layer and output predicted probabilities
output = round(output_proba,0) # round to 1 or 0
output

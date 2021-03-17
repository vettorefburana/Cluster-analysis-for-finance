
##############################################
######### Fit neural network model with Keras
##############################################

source("./feature_preprocessing.R")

#############################
## import features ##########
#############################

# traning e test set 
set.seed(100)
ll <- sample(c(1:nrow(dat_nn)), round(0.9*nrow(dat_nn)), replace = FALSE)
learn <- dat_nn[ll,]
test <- dat_nn[-ll,]

# design matrix (omit Area)
features <- c(14:ncol(dat_nn))
learn.XX <- as.matrix(learn[,features])
test.XX <- as.matrix(test[,features])
lambda.0 <- sum(learn$ClaimNb)/sum(learn$Exposure)

###################################
## network architectures ##########
###################################

# plot of gradient descent performance
plot.loss <- function(loss, title = T){
  
  plot(loss$val_loss,col="blue", type = "l",
       main=if(title == T) list("gradient descent loss") else "",
       xlab="epoch", ylab="loss", 
       lty = 1, lwd = 2,
       ylim=c(min(unlist(loss)), max(unlist(loss))))
  
  lines(loss$loss,col="red", lty = 1, lwd = 2)
}

# shallow neural network
shallow.plain.vanilla <- function(seed, q0, y0, act_fun){
  set.seed(seed)

  design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
  output = design %>%
    layer_dense(units=q0[2], activation=act_fun, name='layer1') %>%
    layer_dense(units=q0[3], activation='exponential', name='output', 
                weights=list(array(0, dim=c(q0[2],q0[3])), array(y0, dim=c(q0[3]))))

  model <- keras_model(inputs = c(design), outputs = c(output))
  model
}

# deep neural network
deep3.plain.vanilla <- function(seed, q0, y0){
  set.seed(seed)
  design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 

  output = design %>%
    layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
    layer_dense(units=q0[3], activation='tanh', name='layer2') %>%
    layer_dense(units=q0[4], activation='tanh', name='layer3') %>%
    layer_dense(units=q0[5], activation='exponential', name='output', 
                weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))

  model <- keras_model(inputs = c(design), outputs = c(output))
  model
}

# deep network with normalization & dropout
deep3.norm.dropout <- function(seed, q0, w0, y0){
  set.seed(seed)

  design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 

  output = design %>%
    layer_dense(units=q0[2], activation='tanh', name='layer1') %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = w0) %>%
    layer_dense(units=q0[3], activation='tanh', name='layer2') %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = w0) %>%
    layer_dense(units=q0[4], activation='tanh', name='layer3') %>%
    layer_dropout(rate = w0) %>%
    layer_dense(units=q0[5], activation='exponential', name='output', 
                weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))

  model <- keras_model(inputs = c(design), outputs = c(output))
  model
}

# deep network with regularization
deep3.ridge <- function(seed, q0, w0, y0){
  set.seed(seed)
 
  design  <- layer_input(shape = c(q0[1]), dtype = 'float32', name = 'design') 
  
  output = design %>%
    layer_dense(units=q0[2], kernel_regularizer=regularizer_l2(w0[1]), activation='tanh', name='layer1') %>%
    layer_dense(units=q0[3], kernel_regularizer=regularizer_l2(w0[2]), activation='tanh', name='layer2') %>%
    layer_dense(units=q0[4], kernel_regularizer=regularizer_l2(w0[3]), activation='tanh', name='layer3') %>%
    layer_dense(units=q0[5], activation='exponential', name='output', 
                weights=list(array(0, dim=c(q0[4],q0[5])), array(y0, dim=c(q0[5]))))

  model <- keras_model(inputs = c(design), outputs = c(output))
  model
}

###################################
## shallow network ##########
###################################

q1 <- 20                               # number of neurons
qqq <- c(length(features), c(q1), 1)   # dimension of all layers including input and output
seed <- 100                            # set seed
epoch0 <- 100 


## choose optimizer and activation function ##########
ottimizzatori = c("sgd", "adagrad", "adadelta", "rmsprop",
                  "adam", "adamax", "nadam")

act_list = c("tanh", "sigmoid")

loss_fin = list()

for (j in 1:length(act_list)){

  model <- shallow.plain.vanilla(seed, qqq, log(lambda.0), act_list[j])

  losses = matrix(0, ncol = 2, nrow = length(ottimizzatori))

  for(i in 1:length(ottimizzatori)){

    model %>% compile(loss = 'poisson', optimizer = ottimizzatori[i])

    fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                         batch_size=10000, epochs=epoch0, verbose=0)

    learn.y <- as.vector(model %>% predict(list(learn.XX)))
    test.y <- as.vector(model %>% predict(list(test.XX)))

    losses[i, ] = c(Poisson.loss(learn.y, learn$ClaimNb),
                    Poisson.loss(test.y, test$ClaimNb))
  }

  colnames(losses) = c("in sample", "out of sample")
  rownames(losses) = ottimizzatori

  loss_fin[[j]] = losses

}

names(loss_fin) = act_list


### choose batch size ###########
sequenza = c(5, 10, 50, 100, 200, 500, 1000)

losses = matrix(0, ncol = 2, nrow = length(sequenza))

fit = list()

for(i in 1:length(sequenza)){
  
  bb = nrow(learn.XX)/sequenza[i]
  
  model <- shallow.plain.vanilla(seed, qqq, log(lambda.0), "tanh")
  
  model %>% compile(loss = 'poisson', optimizer = "nadam")
  
  start_time <- Sys.time()

  fit[[i]] <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=bb, epochs=epoch0, verbose=0)
  
  end_time <- Sys.time()
  
  time_taken = end_time - start_time

  learn.y <- as.vector(model %>% predict(list(learn.XX)))
  test.y <- as.vector(model %>% predict(list(test.XX)))

  losses[i, ] = c(Poisson.loss(learn.y, learn$ClaimNb),
                  Poisson.loss(test.y, test$ClaimNb))
  
}

## choose numbers of neurons ##########

neur = c(5, 10, 20, 30)

losses = matrix(0, ncol = 2, nrow = length(neur))

fit = list()

for(i in 1:length(neur)){
  
  q1 <- neur[i]                         
  qqq <- c(length(features), c(q1), 1)
  
  model <- shallow.plain.vanilla(seed, qqq, log(lambda.0), "tanh")
  
  model %>% compile(loss = 'poisson', optimizer = "nadam")
  
  start_time <- Sys.time()
  
  fit[[i]] <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=600, epochs=epoch0, verbose=0)
  
  end_time <- Sys.time()
  
  time_taken = end_time - start_time
  
  learn.y <- as.vector(model %>% predict(list(learn.XX)))
  test.y <- as.vector(model %>% predict(list(test.XX)))
  
  losses[i, ] = c(Poisson.loss(learn.y, learn$ClaimNb),
                  Poisson.loss(test.y, test$ClaimNb))
  
}

## choose number of epochs ##########

q1 <- 20                        
qqq <- c(length(features), c(q1), 1)

epoch_list = c(100, 200, 500, 1000)

losses = matrix(0, ncol = 2, nrow = length(epoch_list))

fit = list()

for(i in 1:length(epoch_list)){
  
  epoch0 = epoch_list[i]
  
  model <- shallow.plain.vanilla(seed, qqq, log(lambda.0), "tanh")
  
  model %>% compile(loss = 'poisson', optimizer = "nadam")
  
  start_time <- Sys.time()
  
  fit[[i]] <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                            batch_size=600, epochs=epoch0, verbose=0)
  
  end_time <- Sys.time()
  
  time_taken = end_time - start_time
  
  learn.y <- as.vector(model %>% predict(list(learn.XX)))
  test.y <- as.vector(model %>% predict(list(test.XX)))
  
  losses[i, ] = c(Poisson.loss(learn.y, learn$ClaimNb),
                  Poisson.loss(test.y, test$ClaimNb))
  
}


###################################
## deep neural network ##########
###################################

q1 <- c(20,15,10)                      # number of neurons
qqq <- c(length(features), c(q1), 1)   # dimension of all layers including input and output
seed <- 200                            # set seed

## early stopping with callback ################ 

model <- deep3.plain.vanilla(seed, qqq, log(lambda.0))
model

path0 <- paste("./parametri/deep3_plain_vanilla", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", 
                                 verbose = 0,  save_best_only = TRUE, 
                                 save_weights_only = FALSE)

model %>% compile(loss = 'poisson', optimizer = 'nadam')

epoch0 <- 1000
{t1 <- proc.time()
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=6000, epochs=epoch0, verbose=0, callbacks = CBs)
  (proc.time()-t1)[3]}

learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))

losses = c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# results on best validation model (callback)
new_model = load_model_hdf5(path0)

w1 <- get_weights(model)
learn.y <- as.vector(new_model %>% predict(list(learn.XX)))
test.y <- as.vector(new_model %>% predict(list(test.XX)))

# losses
losses = c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

## l2 regularization ############

q1 <- c(20,15,10)                      # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output
seed <- 200                            # set seed

# deep3 network with ridge regularizer
w0 <- rep(0.00001,3)                   #  regularization parameter
model <- deep3.ridge(seed, qqq, w0, log(lambda.0))
model

path0 <- paste("./parametri/deep3_plain_vanilla", sep="")
CBs <- callback_model_checkpoint(path0, monitor = "val_loss", 
                                 verbose = 0,  save_best_only = TRUE, 
                                 save_weights_only = FALSE)

model %>% compile(loss = 'poisson', optimizer = 'nadam')

epoch0 <- 1000  
{t1 <- proc.time()
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                       batch_size=6000, epochs=epoch0, verbose=0, callbacks=CBs)
  (proc.time()-t1)[3]}

learn.y <- as.vector(model %>% predict(list(learn.XX)))
test.y <- as.vector(model %>% predict(list(test.XX)))

losses = c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

# results on best validation model (callback)
new_model = load_model_hdf5(path0)
learn.y <- as.vector(new_model %>% predict(list(learn.XX)))
test.y <- as.vector(new_model %>% predict(list(test.XX)))

losses = c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))

## dropout & normalization ############

q1 <- c(20,15,10)                      # number of neurons
(qqq <- c(length(features), c(q1), 1)) # dimension of all layers including input and output
seed <- 200                            # set seed
epoch0 <- 500

# define deep3 network with normalization layers and dropouts
w0 <- c(0.01, 0.02, 0.05, 0.1)   #  dropout rate

losses_callback = list()
losses = list()

for (i in 1:length(w0)){
  
  model <- deep3.norm.dropout(seed, qqq, 0.05, log(lambda.0))
 
  path0 <- paste("./parametri/deep3_plain_vanilla", sep="")
  CBs <- callback_model_checkpoint(path0, monitor = "val_loss", verbose = 0,  
                                   save_best_only = TRUE, save_weights_only = FALSE)
  
  model %>% compile(loss = 'poisson', optimizer = 'nadam')
  
  fit <- model %>% fit(list(learn.XX), learn$ClaimNb, validation_split=0.1,
                         batch_size=6000, epochs=epoch0, verbose=0, callbacks=CBs)
  
  learn.y <- as.vector(model %>% predict(list(learn.XX)))
  test.y <- as.vector(model %>% predict(list(test.XX)))
 
  losses[[i]] = c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))
  
  # results on best validation model (callback)
  new_model = load_model_hdf5(path0)
  learn.y <- as.vector(new_model %>% predict(list(learn.XX)))
  test.y <- as.vector(new_model %>% predict(list(test.XX)))

  losses_callback[[i]] = c(Poisson.loss(learn.y, learn$ClaimNb),Poisson.loss(test.y, test$ClaimNb))
  
}

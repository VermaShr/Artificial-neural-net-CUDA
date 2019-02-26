%Prableen Kaur kaur0016%

function [z,w,v,t_err,v_err] = mlptrain(train_name, val_name, m, k)

    %getting the data
    train = importdata(train_name,',');
    valid = importdata(val_name,',');
    
    train_labels = train(:,end);
    valid_labels = valid(:,end);
    train = train(:,1:end-1);
    valid = valid(:,1:end-1);
    
    %getting number of inputs
    d = size(train, 2);
    n = size(train, 1);
    
    %one-hot encoding the response r_t
    r_t = zeros(n,k);
    for x = 1:n
        r_t(x,train_labels(x)+1) = 1;
    end
    
    %Initializing the weights of both layers
    %range of initial values
    a = -0.01;
    b = 0.01;
    w = (b-a).*rand(m,d+1) + a;
    v = (b-a).*rand(k,m+1) + a;
    delta_v = zeros(k,m+1);
    delta_w = zeros(m,d+1);
    z = zeros(n,m);
    y = zeros(n,k);
    eta = 0.001;
    flag = 1;
    error_old = 0;
    
    %appending a column of ones to the training data set
    train = [ones(n,1) train];
    
    %Count to keep count of iterations
    %count = 0;
    
    fprintf("Training the Multi Layer Perceptron Model...\n")
    %outer loop of Stochastic gradient descent to repeat until convergence
    while(flag)
        
        %count = count + 1;
        %disp(count);
             
        %For each training data set
        for i = 1:n
            
            %computing hidden layer outputs
            for h = 1:m
                input = w(h,:)*train(i,:)'; 
                if(input < 0)
                    z(i,h) = 0;
                else
                    z(i,h) = input;
                end
            end
            
                       
            %computing outputs
            o = (v(:,2:end)*z(i,:)' + v(:,1))';
            
            %applying softmax
            y(i,:) = exp(o);
            total = sum(y(i,:),2);
            y(i,:) = y(i,:)/total;         
            
            %updates using backpropagation
            for j=1:k
                delta_v(:,2:end) = eta .* ((r_t(i,:) - y(i,:))'.*z(i,:));
                %Adding z_0 the bias term that will give v_i0
                delta_v(:,1) = eta .* ((r_t(i,:) - y(i,:))'*1);
            end
                            
            for h=1:m
                input = w(h,:)*train(i,:)'; 
                if(input < 0)
                    delta_w(h,:) = 0;
                else
                    delta_w(h,:) = eta.*(((r_t(i,:) - y(i,:))* v(:,h+1))'.*train(i,:));                    
                end
            end
            
            w = w + delta_w;
            v = v + delta_v;
            
        end
        %compute convergence criteria - Error function
        error_new = - sum(sum(r_t.*log(y)));
        diff = abs(error_new - error_old);
        
        if(diff < 0.1)
            flag = 0;
        else
            error_old = error_new;
        end        
    end 
    
    %Compute training set errors: 
    %%getting index of max y_i of every training data
    [m_train,pred_labels] = max(y,[],2);
    pred_labels = pred_labels - 1;
    
    train_error = computeError(train_labels,pred_labels);
    
    %Compute Validation set errors
    n_valid = size(valid,1);
    y_valid = zeros(n_valid,k);
    %appending a column of ones to the validation data set
    valid = [ones(n_valid,1) valid];
    
    %%First computing outputs for validation set using learnt model
    %computing hidden layer outputs
    z_valid = valid * w';
    z_valid(z_valid < 0) = 0;
    
    %computing outputs
    o_valid = (v(:,2:end)*z_valid' + v(:,1))';
    
    %applying softmax
    y_valid = exp(o_valid);
    totals = sum(y_valid,2);
    y_valid = y_valid./totals;
    
    %Getting labels and computing error rates
    [m_valid,pred_valid_labels] = max(y_valid,[],2);
    pred_valid_labels = pred_valid_labels - 1;
    
    valid_error = computeError(valid_labels,pred_valid_labels);
    
    %Printing error rates
    fprintf("Error Rates(in %%) for m = %d hidden units: \n",m);
    fprintf("Training Error:\t%f\n",train_error);
    fprintf("Validation Error:\t%f\n",valid_error);
    
    t_err = train_error;
    v_err = valid_error;
end

function err = computeError(true_labels, predicted_labels)
    
    n = size(true_labels,1);
    count = 0;
    
    for i=1:n
        if(true_labels(i) ~= predicted_labels(i))
            count = count + 1;
        end
    end
    
    err = count/n*100;
    
end

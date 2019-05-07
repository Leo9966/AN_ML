function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
temp1 = 0;
for i = 1 : m
  temp1 = temp1 - y(i)*log(sigmoid (theta'*X(i,:)')) - (1 - y(i))*log(1-sigmoid(theta'*X(i,:)'));
end

the_temp1=0;
for i = 2 : size(X,2);
  the_temp1 = the_temp1 + theta(i)*theta(i);
end
J = temp1/m + the_temp1*lambda/(2*m);

for j = 1 : size(theta)
  temp2 = 0;
  for k = 1 : m
    temp2 = temp2 + (sigmoid(theta'*X(k,:)') - y(k))*X(k,j);
  end 
  
  if j==1
    grad(j) = temp2/m ;
  else
    grad(j) = temp2/m + lambda*theta(j)/m;
end





% =============================================================

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = length(theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%h_0 = sigmoid(X(:,1)*theta(1));
##
##J_0 =  (1/m)*sum((-y'*log(h_0)-(1-y)'*log(1-h_0))); % não regularizar theta 1
##
##grad_0 = (1/m)*X'*(h-y); % para j=0 theta1


h = sigmoid(X*theta);
J= (1/m)*sum((-y'*log(h)-(1-y)'*log(1-h))) + (lambda/(2*m))*sumsq(theta(2:n));

%grad_0(1) = (1/m)*X'(1,:)*(h-y);
%grad_0 = (1/m)*sum(h-y);
grad = (1/m)*X'*(h-y);

grad(2:n) = ((1/m)*X'(2:n,:)*(h-y)) + (lambda/m)*theta(2:n); % overwrite with correction

% =============================================================

end

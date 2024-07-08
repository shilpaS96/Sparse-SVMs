function [pred_label, run_time,spars] = l1norm_pin_svm(X_train, Y_train, X_test, Y_test, kernel, tau, C,p1)
start_time = cputime;
m = size(X_train,1);
H = zeros(m,m);
m1=size(X_test,1);
%%
%Kernel Construction
if(kernel==1)
    for i=1:m
        for j=1:m
            H(i,j) = svkernel('linear',X_train(i,:), X_train(j,:), p1);
        end
    end
end

if(kernel==2)
    for i=1:m
        for j=1:m
            H(i,j) = svkernel('rbf',X_train(i,:), X_train(j,:), p1);
        end
    end
end
%%
% Add small amount of zero order regularisation to avoid problems
% when Hessian is badly conditioned.
% H = H+1e-10*eye(size(H));
%%
D =eye(m,m);

for i=1:m
    D(i,i)=Y_train(i);
end

ff= [ones(m,1); ones(m,1);0; C*ones(m,1)];
AA= [-D*H , D*H, D*ones(m,1), -eye(m,m);
     D*H,  -D*H, -D*ones(m,1) -(1/tau)*eye(m,m)];
bb =[-ones(m,1);ones(m,1)];
lb= [zeros(m,1);zeros(m,1);-inf;-inf(m,1)];
soln= linprog(ff,AA,bb,[],[],lb);
p=soln(1:m,1);
q=soln(m+1:2*m,1);
b=soln(2*m+1);
%%
H_test = zeros(m1, m);
if(kernel==1)
    for i=1:m1
        for j=1:m
            H_test(i,j) = svkernel('linear',X_test(i,:), X_train(j,:), p1);
        end
    end
end

if(kernel==2)
    for i=1:m1
        for j=1:m
            H_test(i,j) = svkernel('rbf',X_test(i,:), X_train(j,:), p1);
        end
    end
end
%%
pred_label = sign(H_test*(p-q) +b);
accuracy = (size(Y_test,1) - nnz(pred_label - Y_test))*100 / size(Y_test,1);
spars= length(find((p-q)==0))*100/length(p-q);
run_time = cputime - start_time;
end

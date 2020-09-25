opts = optimoptions('intlinprog','Display','off');

A = Generate_A(num);
b = [1,0,0,-1]';
mu = [10,10,10,10,10,31]';
lb = zeros(length(mu),1);
ub = ones(length(mu),1);
intcon = [1,2,3,4,5,6];

tic
parfor i = 1:1000
    mu = rand(6,1);
    mu(6) = 1.3;
    x = intlinprog(mu,intcon,A,b,[],[],lb,[],opts);
%     x = linprog(mu,A,b,[],[],lb,[],opts);
end
toc

function A = Generate_A(num)
    A = zeros(num+2,2*num+2);
    A(1,1) = 1;
    A(2,1) = -1;
    A(1,2*num+2) = 1;
    A(num+2,2*num+2) = -1;
    for i = 1:num
        A(i+1,2*i) = 1;
        A(i+1,2*i+1) = 1;
        A(i+2,2*i) = -1;
        A(i+2,2*i+1) = -1;
    end
end
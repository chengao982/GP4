clear;
clc;
close all;

opts = optimoptions('intlinprog','Display','off');

tic

num = 2;
% A = [1,0,0,1;
%     -1,1,1,0;
%     0,-1,-1,-1];
D = diag(rand(6,1));
U = orth(rand(6,6));
sigma = U'* D * U;
A = Generate_A(num);
size_A = size(A);
A_idx = [1:size_A(2)];
b = [1,0,0,-1]';
mu = [10,10,10,10,10,30]';
% sigma = [2,-1,1,0;
%         -1,2,0,0;
%         1,0,1,0;
%         0,0,0,1];
N = 100;
r_0 = find(b==1);
r_s = find(b==-1);
iterations = 10;
result = 0;

for iter = 1:iterations
    A_k = A;
    A_idx_k = A_idx;
    b_k = b;
    mu_k = mu;
    sigma_k = sigma;
    r_k = r_0;
    total_cost = 0;

    while (r_k ~= r_s)
        link_to_search = find(A_k(r_k,:)==1);
        values = zeros(1,length(link_to_search));
        nodes = zeros(1,length(link_to_search));
        disp(['Current node is ',num2str(r_k)]);

        for count = 1:length(link_to_search)
            link = link_to_search(count);
            j = find(A_k(:,link)==-1);
            nodes(count) = j;

            dimension = length(mu_k)-1;
            intcon = [1:dimension];
            lb = zeros(dimension,1);
            ub = ones(dimension,1);

            if j == r_s
                values(count) = mu_k(link);

            else
                v_hat = 0;

                A_temp = A_k;
                A_temp(:,link) = [];
                b_temp = b_k;
                b_temp(r_k) = 0;
                b_temp(j) = 1;

                mu_1 = mu_k;
                mu_1(link) = [];
                mu_2 = mu_k(link);

                sigma_11 = sigma_k;
                sigma_11(:,link) = [];
                sigma_11(link,:) = [];
                sigma_12 = sigma_k(:,link);
                sigma_12(link) = [];
                sigma_21 = sigma_k(link,:);
                sigma_21(link) = [];
                sigma_22 = sigma_k(link,link);

                sigma_con = sigma_11-sigma_12*inv(sigma_22)*sigma_21;

                parfor i = 1:N                
                    sample = normrnd(mu_2,sqrt(sigma_22));
                    mu_con = mu_1+sigma_12*inv(sigma_22)*(sample-mu_2);
                    x_temp = intlinprog(mu_con,intcon,[],[],A_temp,b_temp,lb,ub,opts);
                    

                    v_hat = v_hat+x_temp'*mu_con;
                end

                values(count) = mu_2+v_hat/N;
            end
        end

        [value_min,idx_min] = min(values);
        selected_link = link_to_search(idx_min);
        selected_node = nodes(idx_min);

        DISP = [A_idx_k(link_to_search); nodes; values];
        disp(DISP);
        disp(['Selected link is ',num2str(A_idx_k(selected_link)),', whose value is ',num2str(value_min)]);

        A_k(:,selected_link) = [];
        A_idx_k(:,selected_link) = [];
        b_k(r_k) = 0;
        b_k(selected_node) = 1;

        mu_1 = mu_k;
        mu_1(selected_link) = [];
        mu_2 = mu_k(selected_link);

        sigma_11 = sigma_k;
        sigma_11(:,selected_link) = [];
        sigma_11(selected_link,:) = [];
        sigma_12 = sigma_k(:,selected_link);
        sigma_12(selected_link) = [];
        sigma_21 = sigma_k(selected_link,:);
        sigma_21(selected_link) = [];
        sigma_22 = sigma_k(selected_link,selected_link);

        sigma_k = sigma_11-sigma_12*inv(sigma_22)*sigma_21;

        cost = normrnd(mu_2,sqrt(sigma_22));
        mu_k = mu_1+sigma_12*inv(sigma_22)*(cost-mu_2);

        r_k = selected_node;

        total_cost = total_cost+cost;

        disp(['Sampled travel time is ',num2str(cost),', running total cost is ',num2str(total_cost)]);
        disp('--------------------------------------------------------------');
    end

    result = result+total_cost;
end

result = result/iterations;
disp(['Average over ',num2str(iterations),' iterations is ',num2str(result)]);

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
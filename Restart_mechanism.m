function [x,fitx,Par_restart]=Restart_mechanism(x,fitx,Par_restart,I,gbestid,func,current_eval)
Co = 0;
[PopSize,n]=size(x);

for i = 1:PopSize
    if I(i) == 1
        Par_restart.counter(i) = Par_restart.counter(i) + 1;
        Co = Co + Par_restart.counter(i);
    else
        Par_restart.counter(i) = 0;
    end
end
Co;
V_pop = 1;
for i=1:n
    V_pop = V_pop.*abs((max(x(:,i)-min(x(:,i)))))./2;
end
V_pop = sqrt(V_pop);
nVOL = sqrt(V_pop/Par_restart.V_lim);
if nVOL < 0.001
    for i = 1:PopSize
        if Co > 2*PopSize 
            if  Par_restart.counter(i)>2*n && i~=gbestid
                %% 统计�?机的维度
                nDim_j = randi(n);
                nSeq_j=randperm(n);
                j=nSeq_j(1:nDim_j);
                [~,j_size] = size(j);
                %% 统计�?机的列数
                nDim_i = randi(PopSize);
                nSeq_i1=randperm(PopSize);
                j_i1=nSeq_i1(1:nDim_i);
                nSeq_i2=randperm(PopSize);
                j_i2=nSeq_i2(1:nDim_i);
                [~,j_size_i1] = size(j_i1);
                if rand > 0.5
                    for i1=1:j_size_i1
                        x(i,:) = rand * x(j_i1(i1),:) + (1-rand)*(x(j_i2(i1),:)) + rands(1,1)*(x(j_i1(i1),:)-x(j_i2(i1),:));
                        %x(j_i2(i1),:) = rand * x(j_i2(i1),:) + (1-rand)*(x(j_i1(i1),:)) + rands(1,1)*(x(j_i2(i1),:)-x(j_i1(i1),:));
                    end
                else
                    for j1 = 1:j_size
                        j_num = j(1,j1);
                        pop_num = x(:,j_num);
                        pop_num_1 = pop_num(randperm(numel(pop_num),1));
                        pop_num_11 = x(i,j_num);
                        x(i,j_num) = rand * pop_num_11 + (1-rand)*pop_num_1;
                    end
                end
                % fitx(i) = cec17_func(x(i,:)',func);
                Par_restart.counter(i)=0;
                current_eval=current_eval+1;
  
            end
        end
    end
end
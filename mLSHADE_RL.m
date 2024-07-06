%%%%%%%%%%%%%%%%%%%
%% This package is a MATLAB/Octave source code of LSHADE_cnEpSin which is a new version of LSHADE-EpSin.
%% Please see the following papers:
%% 1. LSHADE_cnEpSin:
%%     Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan, Ensemble Sinusoidal Differential Covariance Matrix Adaptation with Euclidean Neighborhood  for Solving CEC2017 Benchmark Problems, in Proc. IEEE Congr. Evol. Comput. CEC 2017, June, Donostia - San Sebastián, Spain

%% 2. LSHADE-EpSin:
%%    Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan and Robert G. Reynolds: An Ensemble Sinusoidal Parameter Adaptation incorporated with L-SHADE for Solving CEC2014 Benchmark Problems, in Proc. IEEE Congr. Evol. Comput. CEC 2016, Canada, July, 2016

%% About L-SHADE, please see following papers:
%% Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.
%%  J. Zhang, A.C. Sanderson: JADE: Adaptive differential evolution with optional external archive,” IEEE Trans Evol Comput, vol. 13, no. 5, pp. 945–958, 2009
%% 3. Restart mechanism: 
%% Song, Zhenghao, and Zhenyu Meng. "Differential Evolution with wavelet basis function based parameter control and dimensional interchange for diversity enhancement." Applied Soft Computing 144 (2023): 110492.

%% 4. Multi operators: 
%% Sallam, Karam M., et al. "Improved multi-operator differential evolution algorithm for solving unconstrained problems." 2020 IEEE congress on evolutionary computation (CEC). IEEE, 2020.


clc;
clear all;

format long;
format compact;

problem_size = 30;

%%% change freq
freq_inti = 0.5;

max_nfes = 10000 * problem_size;

rand('seed', sum(100 * clock));

val_2_reach = 10^(-8);
max_region = 100.0;
min_region = -100.0;
lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];
fhd=@cec17_func;
pb = 0.4;
ps = .5;
n_opr=3;
prob=1./n_opr .* ones(1,n_opr);
S.Ndim = problem_size;
S.Lband = ones(1, S.Ndim)*(-100);
S.Uband = ones(1, S.Ndim)*(100);

%%%% Count the number of maximum generations before as NP is dynamically

G_Max = 2745;

Printing=0;
num_prbs =1;
runs = 25;
run_funcvals = [];

result=zeros(num_prbs,5);

fprintf('Running mLSHADE_RL on D= %d\n', problem_size)
for func = 1:30
    optimum = func * 100.0;
    S.FuncNo = func;

    %% Record the best results
    outcome = [];

    fprintf('\n-------------------------------------------------------\n')
    fprintf('Function = %d, Dimension size = %d\n', func, problem_size)

    %     parfor run_id = 1 : runs
    for run_id = 1 : runs
        run_funcvals = [];
        col=1;              %% to print in the first column in all_results.mat

        %%  parameter settings for L-SHADE
        p_best_rate = 0.11;    %0.11
        arc_rate = 1.4;
        memory_size = 5;
        pop_size = 18 * problem_size;   %18*D
        SEL = round(ps*pop_size);
        prob_ls=0.01;
        max_pop_size = pop_size;
        min_pop_size = 4.0;

        nfes = 0;
        %% Initialize the main population
        popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
        pop = popold; % the old population becomes the current population

        fitness = feval(fhd,pop',func);
        fitness = fitness';

        bsf_fit_var = 1e+30;
        bsf_index = 0;
        bsf_solution = zeros(1, problem_size);

        %%%%%%%%%%%%%%%%%%%%%%%% for in
        for i = 1 : pop_size
            nfes = nfes + 1;

            if fitness(i) < bsf_fit_var
                bsf_fit_var = fitness(i);
                bsf_solution = pop(i, :);
                bsf_index = i;
            end

            if nfes > max_nfes; break; end
        end
         %%%%%%%%%%%%%%%%%%%%%%%% for out
        best_fit_var=min(fitness);
        Best_Func_Val=repmat(best_fit_var,1,pop_size);
       

        memory_sf = 0.5 .* ones(memory_size, 1);
        memory_cr = 0.5 .* ones(memory_size, 1);

        memory_freq = freq_inti*ones(memory_size, 1);
        memory_pos = 1;

        archive.NP = arc_rate * pop_size; % the maximum size of the archive
        archive.pop = zeros(0, problem_size); % the solutions stored in te archive
        archive.funvalues = zeros(0, 1); % the function value of the archived solutions

        gg=0;  %%% generation counter used For Sin
        igen =1;  %%% generation counter used For LS

        flag1 = false;
        flag2 = false;

        goodF1all = [];
        goodF2all =[];
        badF1all = [];
        badF2all = [];
        goodF1 = [];
        goodF2 = [];
        badF1 = [];
        badF2 = [];
        Par_restart.lu = lu;
        Par_restart.V_lim = 1;
        for i=1:problem_size
            Par_restart.V_lim = (Par_restart.V_lim .*(abs(Par_restart.lu(1,i) - Par_restart.lu(2,i))));
        end
        Par_restart.V_lim = sqrt(Par_restart.V_lim);
        Par_restart.counter = zeros(pop_size,1);
        Par_restart.m = 1;
        Par_restart.m_max = 4;
        Par_restart.m_min = 1;
             %% main loop
        while nfes <= max_nfes
            gg=gg+1;

            pop = popold; % the old population becomes the current population
            [temp_fit, sorted_index] = sort(fitness, 'ascend');

            mem_rand_index = ceil(memory_size * rand(pop_size, 1));
            mu_sf = memory_sf(mem_rand_index);
            mu_cr = memory_cr(mem_rand_index);
            mu_freq = memory_freq(mem_rand_index);

            %% for generating crossover rate
            cr = normrnd(mu_cr, 0.1);
            term_pos = find(mu_cr == -1);
            cr(term_pos) = 0;
            cr = min(cr, 1);
            cr = max(cr, 0);

            %% for generating scaling factor
            sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
            pos = find(sf <= 0);

            while ~ isempty(pos)
                sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                pos = find(sf <= 0);
            end


            freq = mu_freq + 0.1 * tan(pi*(rand(pop_size, 1) - 0.5));
            pos_f = find(freq <=0);
            while ~ isempty(pos_f)
                freq(pos_f) = mu_freq(pos_f) + 0.1 * tan(pi * (rand(length(pos_f), 1) - 0.5));
                pos_f = find(freq <= 0);
            end

            sf = min(sf, 1);
            freq = min(freq, 1);

            LP = 20;
            flag1 = false;
            flag2 = false;
            if(nfes <= max_nfes/2)
                flag1 = false;
                flag2 = false;
                if (gg <= LP)
                    %% Both have the same probability
                    %% Those generations are the learning period
                    %% Choose one of them randomly
                    p1 = 0.5;
                    p2 = 0.5;
                    c=rand;
                    if(c < p1)
                        sf = 0.5.*( sin(2.*pi.*freq_inti.*gg+pi) .* ((G_Max-gg)/G_Max) + 1 ) .* ones(pop_size,problem_size);
                        flag1 = true;
                    else
                        sf = 0.5 *( sin(2*pi .* freq(:, ones(1, problem_size)) .* gg) .* (gg/G_Max) + 1 ) .* ones(pop_size,problem_size);
                        flag2 = true;
                    end

                else
                    %% compute the probability as used in SaDE
                    ns1 = size(goodF1,1);
                    ns1_sum = 0;
                    nf1_sum = 0;
                    %               for hh = 1 : size(goodF1all,2)
                    for hh = gg-LP : gg-1
                        ns1_sum = ns1_sum + goodF1all(1,hh);
                        nf1_sum = nf1_sum + badF1all(1,hh);
                    end
                    sumS1 = (ns1_sum/(ns1_sum + nf1_sum)) + 0.01;


                    ns2 = size(goodF2,1);
                    ns2_sum = 0;
                    nf2_sum = 0;
                    %             for hh = gg-LP : gg-1
                    %               for hh = 1 : size(goodF2all,2)
                    for hh = gg-LP : gg-1
                        ns2_sum = ns2_sum + goodF2all(1,hh);
                        nf2_sum = nf2_sum + badF2all(1,hh);
                    end
                    sumS2 = (ns2_sum/(ns2_sum + nf2_sum)) + 0.01;

                    p1 = sumS1/(sumS1 + sumS2);
                    p2 = sumS2/(sumS2 + sumS1);

                    if(p1 > p2)
                        sf = 0.5.*( sin(2.*pi.*freq_inti.*gg+pi) .* ((G_Max-gg)/G_Max) + 1 ) .* ones(pop_size,problem_size);
                        flag1 = true;
                        %                   size(goodF1,1)
                    else
                        sf = 0.5 *( sin(2*pi .* freq(:, ones(1, problem_size)) .* gg) .* (gg/G_Max) + 1 ) .* ones(pop_size,problem_size);
                        flag2 = true;
                        %                   size(goodF2,1)
                    end
                end
            end

            %% mutation
            bb= rand(pop_size, 1);
            probiter = prob(1,:);
            l2= sum(prob(1:2));
            op_1 = bb <=  probiter(1)*ones(pop_size, 1);
            op_2 = bb > probiter(1)*ones(pop_size, 1) &  bb <= (l2*ones(pop_size, 1)) ;
            op_3 = bb > l2*ones(pop_size, 1) &  bb <= (ones(pop_size, 1)) ;

            r0 = [1 : pop_size];
            popAll = [pop; archive.pop];
            [r1, r2, r3] = gnR1R2R3(pop_size, size(popAll, 1), r0);

            pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
            randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]
            randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
            pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions
            if(nfes <= 0.2*max_nfes)
                FW=0.7*sf;
            elseif(nfes > 0.2*max_nfes)&& (nfes <= 0.4*max_nfes)
                FW=0.8*sf;
            else
                FW=1.2*sf;
            end

            vi=zeros(pop_size,problem_size);

            %% Multi-operators

            % DE/current-to-pbest/1 with archive
            vi(op_1==1,:) = pop(op_1==1,:)+ FW(op_1==1, ones(1, problem_size)) .*(pbest(op_1==1,:) - pop(op_1==1,:) + pop(r1(op_1==1), :) - popAll(r2(op_1==1), :));

            % DE/current-to-pbest/1 without archive
            vi(op_2==1,:) =  pop(op_2==1,:)+ sf(op_2==1, ones(1, problem_size)) .*(pbest(op_2==1,:) - pop(op_2==1,:) + pop(r1(op_2==1), :) - pop(r3(op_2==1), :));

            % DE/current-to-ordpbest-weight/1
            EDEpNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
            EDErandindex = ceil(rand(1, pop_size) .* EDEpNP); %% select from [1, 2, 3, ..., pNP]
            EDErandindex = max(1, EDErandindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
            EDEpestind=sorted_index(EDErandindex);
            R1 = Gen_R(pop_size,2);
            R1(:,1)=[];
            R1=[R1 EDEpestind];
            fr=fitness(R1);
            [~,I1] = sort(fr,2);
            R_S=[];
            for i=1:pop_size
                R_S(i,:)=R1(i,I1(i,:));
            end
            rb=R_S(:,1)';
            rm=R_S(:,2)';
            rw=R_S(:,3)';
            vi(op_3==1,:) = pop(op_3==1,:) + FW(op_3==1, ones(1, problem_size)) .* (pop(rb(op_3==1), :) - pop(op_3==1,:) + pop(rm(op_3==1), :) - popAll(rw(op_3==1), :));
            vi = boundConstraint(vi, pop, lu);

            %       %% Bin Crx
            %       mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent
            %       rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
            %       jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
            %       ui = vi; ui(mask) = pop(mask);
            %       %%

            %% Bin crossover according to the Eigen coordinate system
            J_= mod(floor(rand(pop_size, 1)*problem_size), problem_size) + 1;
            J = (J_-1)*pop_size + (1:pop_size)';
            crs = rand(pop_size, problem_size) < cr(:, ones(1, problem_size));
            if rand<pb
                %% coordinate ratation

                %%%%% Choose neighbourhood region to the best individual
                best = pop(sorted_index(1), :);
                Dis = pdist2(pop,best,'euclidean'); % euclidean distance
                %D2 = sqrt(sum((pop(1,:) - best).^2, 2));

                %%%% Sort
                [Dis_ordered idx_ordered] = sort(Dis, 'ascend');
                SEL;
                Neighbour_best_pool = pop(idx_ordered(1:SEL), :); %%% including best also so start from 1
                Xsel = Neighbour_best_pool;
                %            sizz = size(Xsel)
                %%%%%%%%%%%%%%%%%%%%%%%%%%%


                %Xsel = pop(sorted_index(1:SEL), :);
                xmean = mean(Xsel);
                % covariance matrix calculation
                C =  1/(SEL-1)*(Xsel - xmean(ones(SEL,1), :))'*(Xsel - xmean(ones(SEL,1), :));
                C = triu(C) + transpose(triu(C,1)); % enforce symmetry
                [R,D] = eig(C);
                % limit condition of C to 1e20 + 1
                if max(diag(D)) > 1e20*min(diag(D))

                    tmp = max(diag(D))/1e20 - min(diag(D));
                    C = C + tmp*eye(problem_size);
                    [R, D] = eig(C);
                end
                TM = R;
                TM_=R';
                Xr = pop*TM;
                vi = vi*TM;
                %% crossover according to the Eigen coordinate system
                Ur = Xr;
                Ur(J) = vi(J);
                Ur(crs) = vi(crs);
                %%
                ui = Ur*TM_;

            else

                ui = pop;
                ui(J) = vi(J);
                ui(crs) = vi(crs);

            end
            %%%%%%%%

            children_fitness = feval(fhd, ui', func);
            children_fitness = children_fitness';


            %%%% To check stagnation
            flag = false;
            bsf_fit_var_old = bsf_fit_var;
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            for i = 1 : pop_size
                % nfes = nfes + 1;

                if children_fitness(i) < bsf_fit_var
                    bsf_fit_var = children_fitness(i);
                    bsf_solution = ui(i, :);
                    bsf_index = i;
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%% for out

            dif = abs(fitness - children_fitness);


            %% I == 1: the parent is better; I == 2: the offspring is better
            I = (fitness > children_fitness);
            goodCR = cr(I == 1);
            goodF = sf(I == 1);
            goodFreq = freq(I == 1);
            dif_val = dif(I == 1);

            %% change here also
            %% recored bad too
            badF = sf(I == 0);
            if flag1 == true
                goodF1 = goodF;
                goodF1all = [goodF1all size(goodF1,1)];

                badF1 = badF;
                badF1all = [badF1all size(badF1,1)];

                %% Add zero for other one  or add 1 to prevent the case of having NaN
                goodF2all = [goodF2all 1];
                badF2all = [badF2all 1];

            end
            if flag2 == true
                goodF2 = goodF;
                goodF2all = [goodF2all size(goodF2,1)];

                badF2 = badF;
                badF2all = [badF2all size(badF2,1)];

                %% Add zero for other one
                goodF1all = [goodF1all 1];
                badF1all = [badF1all 1];
            end

            %% ==================== update Prob. of each DE operators ===========================
            diff2 = max(0,(fitness - children_fitness))./abs(children_fitness);
            count_S(1)=max(0,mean(diff2(op_1==1)));
            count_S(2)=max(0,mean(diff2(op_2==1)));
            count_S(3)=max(0,mean(diff2(op_3==1)));

            %% update probs.
            if count_S~=0
                prob= max(0.1,min(0.9,count_S./(sum(count_S))));
            else
                prob=1/3 * ones(1,3);
            end

            %      isempty(popold(I == 1, :))
            archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));

            [fitness, I] = min([fitness, children_fitness], [], 2);

            run_funcvals = [run_funcvals; fitness];

            popold = pop;
            popold(I == 2, :) = ui(I == 2, :);

            num_success_params = numel(goodCR);

            if num_success_params > 0
                sum_dif = sum(dif_val);
                dif_val = dif_val / sum_dif;

                %% for updating the memory of scaling factor
                memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);

                %% for updating the memory of crossover rate
                if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                    memory_cr(memory_pos)  = -1;
                else
                    memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                end

                %% for updating the memory of freq
                if max(goodFreq) == 0 || memory_freq(memory_pos)  == -1
                    memory_freq(memory_pos)  = -1;
                else
                    memory_freq(memory_pos) = (dif_val' * (goodFreq .^ 2)) / (dif_val' * goodFreq);
                end

                memory_pos = memory_pos + 1;
                if memory_pos > memory_size;  memory_pos = 1; end
            end
            %% Restart mechanism
            [~, bes_l]=min(fitness);
            [popold,fitness,Par_restart]=Restart_mechanism(popold,fitness,Par_restart,I,bes_l,func,nfes);

            %% for resizing the population size
            plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);

            if pop_size > plan_pop_size
                reduction_ind_num = pop_size - plan_pop_size;
                if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end

                pop_size = pop_size - reduction_ind_num;
                SEL = round(ps*pop_size);
                for r = 1 : reduction_ind_num
                    [valBest indBest] = sort(fitness, 'ascend');
                    worst_ind = indBest(end);
                    popold(worst_ind,:) = [];
                    pop(worst_ind,:) = [];
                    fitness(worst_ind,:) = [];
                end

                archive.NP = round(arc_rate * pop_size);

                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1 : archive.NP);
                    archive.pop = archive.pop(rndpos, :);
                end
            end
            [bsf_fit_var,indx]=min(fitness);
            bestx=popold(indx,:);
            %% ============================ LS2 ====================================
            if nfes>0.85*max_nfes
                if rand<prob_ls

                    [bestx,bestold,~,succ] = LS2 (bestx,bsf_fit_var,nfes,func,max_nfes,lu(1,:),lu(2,:));
                    if succ==1 %% if LS2 was successful

                        popold(pop_size,:)=bestx';
                        fitness(pop_size)=bestold;
                        [fitness, sort_indx]=sort(fitness);
                        popold= popold(sort_indx,:);
                        prob_ls=0.1;
                    else
                        prob_ls=0.01; %% set p_LS to a small value it  LS was not successful
                    end

                end

            end

            [bsf_fit_var,~]=min(fitness);
            bsf_fit_var_old=bsf_fit_var;
            Best_Func_Val=[Best_Func_Val repmat(bsf_fit_var,1,pop_size)];
            nfes=nfes+pop_size;
        end %% End nfes

        bsf_error_val = bsf_fit_var - optimum;
        if bsf_error_val < val_2_reach
            bsf_error_val = 0;
        end

        fprintf('%d th run, best-so-far error value = %1.8e\n', run_id , bsf_error_val)
        outcome = [outcome bsf_error_val];
        res_val1=Best_Func_Val(1:max_nfes)- optimum;
      

    end %% end 2 run

    %     fprintf('\n')
    %     fprintf('min error value = %1.8e, max = %1.8e, median = %1.8e, mean = %1.8e, std = %1.8e\n', min(outcome), max(outcome), median(outcome), mean(outcome), std(outcome))

    result(func,1)=  min(outcome);
    result(func,2)=  max(outcome);
    result(func,3)=  median(outcome);
    result(func,4)=  mean(outcome);
    result(func,5)=  std(outcome);
    Final_results= [min(outcome),max(outcome),median(outcome), mean(outcome),std(outcome)];



    if Printing==1
        save('mLSAHDE_LR_results_30D.csv', 'Final_results', '-ascii','-append');
        lim=10*problem_size:10*problem_size:max_nfes;
        res_to_print=res_val1(:,lim);
        name1 = 'Results_Record_mLSHADE_LR\mLSAHDE_LR_F#';
        name2 = num2str(func);
        name3 = '_D#';
        name4 = num2str(problem_size);
        name5 = '.mat';
        f_name=strcat(name1,name2,name3,name4,name5);
        res_to_print=res_to_print';
        save(f_name, 'res_to_print1', '-ascii');
        name5 = '.dat';
        f_name=strcat(name1,name2,name3,name4,name5);
        save(f_name, 'res_to_print', '-ascii');

    end
end %% end 1 function run

%%
% File Name: AdaBoost
% This is the implementation of the ada boost algorithm.
% Parameters - very easy to gues by name...
% Return values: i - hypothesis-index  vector.
%                t - threshhols vector
%                beta - weighted beta.
%%
function boosted=adaBoost(train,train_label,cycles)
    disp('running adaBoost algorithm');
    d=size(train);
	distribution=ones(1,d(1))/d(1); %u
	error=zeros(1,cycles);
	beta=zeros(1,cycles); 
	label=(train_label(:)>=5);% contain the correct label per vector
	for j=1:cycles
        %if(mod(j,10)==0)
        %    disp([j,cycles]);
        %end
	    [i,t]=weakLearner(distribution,train,label);
        error(j)=distribution*abs(label-(train(:,i)>=t));
        beta(j)=error(j)/(1-error(j));
        boosted(j,:)=[beta(j),i,t];
        
        distribution=distribution.* exp(log(beta(j))*(1-abs(label-(train(:,i)>=t ))))'; 
        distribution=distribution/sum(distribution); %normalize to 0~1
        if j < 4
            fprintf("%d turns\n, featureID=%d, theta=%d\n",j,i,t);
            fprintf("blending weight = %f\n",log(beta(jX)));
        end
  
end
    
    